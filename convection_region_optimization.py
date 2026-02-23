import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
from util import *
from model_dict import get_model

parser = argparse.ArgumentParser('Training Region Optimization')
parser.add_argument('--model', type=str, default='pinn')
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--initial_region', type=float, default=1e-4)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--past_iterations', type=int, default=5)
parser.add_argument('--max_iters', type=int, default=1000)
parser.add_argument('--sampling_mode', type=str, default='one_sided', choices=['one_sided', 'symmetric'])
parser.add_argument('--residual_loss', type=str, default='mse', choices=['mse', 'huber'])
parser.add_argument('--huber_delta', type=float, default=0.05)
parser.add_argument('--ff_dim', type=int, default=64)
parser.add_argument('--ff_scale', type=float, default=1.0)
parser.add_argument('--ff_scale_x', type=float, default=None)
parser.add_argument('--ff_scale_t', type=float, default=None)
parser.add_argument('--ff_scale_char', type=float, default=None)
parser.add_argument('--use_characteristic', action='store_true')
parser.add_argument('--adv_speed', type=float, default=50.0)
parser.add_argument('--char_aligned_sampling', action='store_true')
parser.add_argument('--sample_time_scale', type=float, default=1.0)
parser.add_argument('--sample_ortho_scale', type=float, default=0.0)
parser.add_argument('--periodic_x_sampling', action='store_true')
parser.add_argument('--w_res', type=float, default=1.0)
parser.add_argument('--w_bc', type=float, default=1.0)
parser.add_argument('--w_ic', type=float, default=1.0)
parser.add_argument('--run_tag', type=str, default='')
parser.add_argument('--paper_outputs', action='store_true')
args = parser.parse_args()

seed = int(args.seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = resolve_device(args.device)
run_tag = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in args.run_tag.strip())


def tagged_path(path):
    if not run_tag:
        return path
    root, ext = os.path.splitext(path)
    return f'{root}_{run_tag}{ext}'

x_min, x_max = 0.0, 2 * np.pi
t_min, t_max = 0.0, 1.0

res, b_left, b_right, b_upper, b_lower = get_data([x_min, x_max], [t_min, t_max], 101, 101)
res_test, _, _, _, _ = get_data([x_min, x_max], [t_min, t_max], 101, 101)

if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    res = make_time_sequence(res, num_step=5, step=1e-4)
    b_left = make_time_sequence(b_left, num_step=5, step=1e-4)
    b_right = make_time_sequence(b_right, num_step=5, step=1e-4)
    b_upper = make_time_sequence(b_upper, num_step=5, step=1e-4)
    b_lower = make_time_sequence(b_lower, num_step=5, step=1e-4)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)


def wrap_periodic(x, x_range):
    x_min, x_max = x_range
    width = x_max - x_min
    if width <= 0:
        return x
    return torch.remainder(x - x_min, width) + x_min


def sample_region_points(
    x_base,
    t_base,
    region_radius,
    sample_num,
    sampling_mode,
    x_range,
    t_range,
    char_aligned_sampling=False,
    adv_speed=50.0,
    sample_time_scale=1.0,
    sample_ortho_scale=0.0,
    periodic_x_sampling=False,
):
    x_samples = []
    t_samples = []
    for _ in range(sample_num):
        if char_aligned_sampling:
            if sampling_mode == 'symmetric':
                t_noise = (2.0 * torch.rand_like(t_base) - 1.0) * (region_radius * sample_time_scale)
            else:
                t_noise = torch.rand_like(t_base) * (region_radius * sample_time_scale)
            x_noise = adv_speed * t_noise
            if sample_ortho_scale > 0:
                x_noise = x_noise + (2.0 * torch.rand_like(x_base) - 1.0) * (region_radius * sample_ortho_scale)
        else:
            if sampling_mode == 'symmetric':
                x_noise = (2.0 * torch.rand_like(x_base) - 1.0) * region_radius
                t_noise = (2.0 * torch.rand_like(t_base) - 1.0) * region_radius
            else:
                x_noise = torch.rand_like(x_base) * region_radius
                t_noise = torch.rand_like(t_base) * region_radius
        x_raw = x_base + x_noise
        if periodic_x_sampling:
            x_samples.append(wrap_periodic(x_raw, x_range))
        else:
            x_samples.append(torch.clamp(x_raw, min=x_range[0], max=x_range[1]))
        t_samples.append(torch.clamp(t_base + t_noise, min=t_range[0], max=t_range[1]))
    return torch.cat(x_samples, dim=0), torch.cat(t_samples, dim=0)


def flatten_gradients(model, device):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros(1, device=device))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                  noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=2).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    model.apply(init_weights)
elif args.model == 'PINN_ResFF':
    model = get_model(args).Model(
        in_dim=2,
        hidden_dim=512,
        out_dim=1,
        num_layer=4,
        ff_dim=args.ff_dim,
        ff_scale=args.ff_scale,
        ff_scale_x=args.ff_scale_x,
        ff_scale_t=args.ff_scale_t,
        ff_scale_char=args.ff_scale_char,
        use_characteristic=args.use_characteristic,
        adv_speed=args.adv_speed,
    ).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)

optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(get_n_params(model))
loss_track = []

# for region optimization
initial_region = args.initial_region
sample_num = max(1, args.sample_num)
past_iterations = args.past_iterations
gradient_list_overall = []
gradient_list_temp = []
gradient_variance = 1

for i in tqdm(range(args.max_iters)):
    ###### Region Optimization with Monte Carlo Approximation ######
    def closure():
        region_radius = np.clip(initial_region / gradient_variance, a_min=0, a_max=0.01)
        x_res_region_sample, t_res_region_sample = sample_region_points(
            x_res,
            t_res,
            region_radius=region_radius,
            sample_num=sample_num,
            sampling_mode=args.sampling_mode,
            x_range=(x_min, x_max),
            t_range=(t_min, t_max),
            char_aligned_sampling=args.char_aligned_sampling,
            adv_speed=args.adv_speed,
            sample_time_scale=max(float(args.sample_time_scale), 1e-6),
            sample_ortho_scale=max(float(args.sample_ortho_scale), 0.0),
            periodic_x_sampling=args.periodic_x_sampling,
        )
        pred_res = model(x_res_region_sample, t_res_region_sample)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = \
            torch.autograd.grad(pred_res, x_res_region_sample, grad_outputs=torch.ones_like(pred_res),
                                retain_graph=True,
                                create_graph=True)[0]
        u_t = \
            torch.autograd.grad(pred_res, t_res_region_sample, grad_outputs=torch.ones_like(pred_res),
                                retain_graph=True,
                                create_graph=True)[0]

        residual = u_t + args.adv_speed * u_x
        if args.residual_loss == 'huber':
            loss_res = F.huber_loss(
                residual,
                torch.zeros_like(residual),
                delta=args.huber_delta,
                reduction='mean',
            )
        else:
            loss_res = torch.mean(residual ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:, 0] - torch.sin(x_left[:, 0])) ** 2)

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

        loss = args.w_res * loss_res + args.w_bc * loss_bc + args.w_ic * loss_ic
        optim.zero_grad()
        loss.backward(retain_graph=True)
        gradient_list_temp.append(flatten_gradients(model, device=device).detach().cpu().numpy())
        return loss


    optim.step(closure)

    ###### Trust Region Calibration ######
    gradient_list_overall.append(np.mean(np.array(gradient_list_temp), axis=0))
    gradient_list_overall = gradient_list_overall[-past_iterations:]
    gradient_list = np.array(gradient_list_overall)
    gradient_variance = (np.std(gradient_list, axis=0) / (
            np.mean(np.abs(gradient_list), axis=0) + 1e-6)).mean()  # normalized variance
    gradient_list_temp = []
    if gradient_variance == 0:
        gradient_variance = 1  # for numerical stability

print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

if not os.path.exists('./results/'):
    os.makedirs('./results/')
torch.save(model.state_dict(), tagged_path(f'./results/convection_{args.model}_region.pt'))

if args.paper_outputs:
    loss_arr = np.array(loss_track, dtype=np.float64)
    total_loss = np.sum(loss_arr, axis=1, keepdims=True)
    loss_dump = np.concatenate([loss_arr, total_loss], axis=1)
    np.savetxt(
        tagged_path(f'./results/convection_{args.model}_region_loss.csv'),
        loss_dump,
        delimiter=',',
        header='loss_0,loss_1,loss_2,total_loss',
        comments='',
    )

    plt.figure(figsize=(5, 3))
    plt.plot(loss_dump[:, -1], color='tab:blue')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss (log)')
    plt.title('Training Loss Curve')
    plt.tight_layout()
    plt.savefig(tagged_path(f'./results/convection_{args.model}_region_optimization_loss.pdf'), bbox_inches='tight')

# Visualize
if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    res_test = make_time_sequence(res_test, num_step=5, step=1e-4)

res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1]
    pred = pred.cpu().detach().numpy()

pred = pred.reshape(101, 101)


def u_res(x, t):
    print(x.shape)
    print(t.shape)
    return np.sin(x - args.adv_speed * t)


res_test, _, _, _, _ = get_data([x_min, x_max], [t_min, t_max], 101, 101)
u = u_res(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

if args.paper_outputs:
    with open(tagged_path(f'./results/convection_{args.model}_region_metrics.csv'), 'w', encoding='utf-8') as fp:
        fp.write('metric,value\n')
        fp.write(f'relative_l1,{rl1:.10f}\n')
        fp.write(f'relative_l2,{rl2:.10f}\n')
        fp.write(f'loss_0,{loss_track[-1][0]:.10f}\n')
        fp.write(f'loss_1,{loss_track[-1][1]:.10f}\n')
        fp.write(f'loss_2,{loss_track[-1][2]:.10f}\n')
        fp.write(f'train_loss,{np.sum(loss_track[-1]):.10f}\n')

plt.figure(figsize=(4, 3))
plt.imshow(pred, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(tagged_path(f'./results/convection_{args.model}_region_optimization_pred.pdf'), bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(tagged_path('./results/convection_exact.pdf'), bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, aspect='equal', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(tagged_path(f'./results/convection_{args.model}_region_optimization_error.pdf'), bbox_inches='tight')
