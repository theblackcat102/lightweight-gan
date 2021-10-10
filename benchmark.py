import torch


N_RUN = 10


if __name__ == "__main__":
    a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
    b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')

    avg_time = 0
    for _ in range(N_RUN):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        ab_full = a_full @ b_full

        end_event.record()
        torch.cuda.synchronize()     
        avg_time += start_event.elapsed_time(end_event)

    print(avg_time/N_RUN)
    mean = ab_full.abs().mean()  
    # pytorch doc : 80.7277 on GA100 ?
    # takes 3.9627s on RTX 3090


    a = a_full.float()
    b = b_full.float()

    # Do matmul at TF32 mode.

    avg_time = 0
    for _ in range(N_RUN):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        ab_tf32 = a @ b  
        # pytorch doc : takes 0.016s on GA100
        # takes 0.0593s on RTX 3090

        end_event.record()
        torch.cuda.synchronize()     
        avg_time += start_event.elapsed_time(end_event)

    print(avg_time/N_RUN)
    error = (ab_tf32 - ab_full).abs().max()  # 0.1747
    relative_error = error / mean  # 0.0022



    torch.backends.cuda.matmul.allow_tf32 = False
    avg_time = 0
    for _ in range(N_RUN):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        ab_fp32 = a @ b  
        # pytorch doc: takes 0.11s on GA100
        # takes 0.1049s on RTX 3090

        end_event.record()
        torch.cuda.synchronize()     
        avg_time += start_event.elapsed_time(end_event)

    print(avg_time/N_RUN)
    error = (ab_fp32 - ab_full).abs().max()  # 0.0031
    relative_error = error / mean  # 0.000039