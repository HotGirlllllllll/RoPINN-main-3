python 1d_wave_region_optimization.py --model PINN --device 'auto' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_wave_region_optimization.py --model QRes --device 'auto' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_wave_region_optimization.py --model FLS --device 'auto' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_wave_region_optimization.py --model KAN --device 'auto' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
python 1d_wave_region_optimization.py --model PINNsFormer --device 'auto' --sampling_mode symmetric --residual_loss huber --huber_delta 0.05
