python 1d_reaction_region_optimization.py --model PINN --device 'cuda:0' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_reaction_region_optimization.py --model QRes --device 'cuda:0' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_reaction_region_optimization.py --model FLS --device 'cuda:0' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_reaction_region_optimization.py --model KAN --device 'cuda:0' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_reaction_region_optimization.py --model PINNsFormer --device 'cuda:0' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
