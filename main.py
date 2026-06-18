from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim

from src.blocks import UNet
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=1, type=int, choices=(0, 1))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load data from https://www.openml.org/d/554
    # (70000, 784) values between 0-255
    from torchvision import datasets
    import torchvision.transforms as transforms
    
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import torch.utils.data as data_utils

    # Select training_set and testing_set
    transform =  transforms.Compose([transforms.Resize(32), transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    # train_loader = datasets.MNIST("data", 
    #                               train= True,
    #                              download=True,
    #                              transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=60000,
    #                                             shuffle=True, num_workers=0)

    test_loader = datasets.MNIST("data", 
                                  train= False,
                                 download=True,
                                 transform=transform)

    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=10000,
                                                shuffle=True, num_workers=0)

    # x = torch.cat([next(iter(test_loader))[0],next(iter(train_loader))[0]],0)
    x = next(iter(test_loader))[0]
    x = x.view(-1,32*32).numpy()
    # x = torch.squeeze(x,1).numpy()

    # print(x.shape)
    # print(torch.min(x))

    # for data, target in test_loader:
    #     print(data.shape)

    # exit()



    # x, _ = fetch_openml("mnist_784") # , version=1, return_X_y=True, as_frame=False, cache=True)

    # # Reshape to 32x32
    # x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
    # x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
    # x = rearrange(x, "b h w -> b (h w)")

    # # Standardize to [-1, 1]
    # input_mean = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # input_sd = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # x = ((x - input_mean) / input_sd).astype(np.float32)

    nn_module = UNet(1, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(1, 32, 32,),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
        ),
    )
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    if args.load_trained:
        model.load_state_dict(torch.load("./ckpts/mnist_trained.pt",map_location=torch.device(args.device)))
    else:
        for step_num in range(args.iterations):
            x_batch = x[np.random.choice(len(x), args.batch_size)]
            x_batch = torch.from_numpy(x_batch).to(args.device)
            x_batch = rearrange(x_batch, "b (h w) -> b () h w", h=32, w=32)
            optimizer.zero_grad()
            loss = model.loss(x_batch).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step_num % 100 == 0:
                logger.info(f"Iter: {step_num}\t" + f"Loss: {loss.data:.2f}\t")
        torch.save(model.state_dict(), "./ckpts/mnist_trained.pt")

    model.eval()

    input_sd = 127
    input_mean = 127
    x_vis = x[:32] * input_sd + input_mean

    ##################
    # define here your degraded images as deg_x, e.g.,

    x_true = x[:32].reshape(32,1,32,32).copy()

    deg_x = 0.7071 * (x_true + np.random.randn(32,1,32,32).astype(np.float32))
    noise = 1.0

    # end of your code
    ##################

    samples = model.sample(bsz=32, noise = noise, x0 = deg_x, device=args.device, stochastic=False, stochastic_noise_scale=1.0).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean

    nrows, ncols = 10, 3
    percents = min(len(samples),4)

    raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

    deg_x = deg_x * input_sd + input_mean
    
    # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images. 
    # First rows show the denoising progression
    for percent_idx in range(percents):
        itr_num = int(round(percent_idx / (percents-1) * (len(samples)-1)))
        print(itr_num)
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx)
            raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = samples[itr_num][i].reshape(32, 32)

        # last block of nrow,ncol of input images
    for i in range(nrows * ncols):
        offset = 32 * ncols * percents
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

    for i in range(nrows * ncols):
        offset =  32 * ncols * (percents+1)
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = deg_x[i].reshape(32, 32)

    raster[:,::32*3] = 64

    plt.imsave("./examples/ex_mnist.png", raster, vmin=0, vmax=255, cmap='gray')

    # ##############################
    # # Distribution Analysis
    # ##############################
    #
    # import os
    # import csv
    #
    # os.makedirs("./examples", exist_ok=True)
    #
    # def make_degraded_images(x_clean, noise_level=1.0, seed=0):
    #     """
    #     x_clean: numpy array of shape (N, 1, 32, 32), values in normalized scale [-1, 1].
    #
    #     Returns noisy/degraded images with the same scale convention.
    #     """
    #     rng = np.random.default_rng(seed)
    #     z = rng.standard_normal(x_clean.shape).astype(np.float32)
    #
    #     # For noise_level = 1:
    #     # deg_x = (x + z) / sqrt(2)
    #     deg_x = (x_clean + noise_level * z) / np.sqrt(1.0 + noise_level ** 2)
    #
    #     return deg_x.astype(np.float32)
    #
    #
    # def compute_metric_values(images):
    #     """
    #     images: numpy array of shape (N, 1, 32, 32), values in normalized scale [-1, 1].
    #
    #     Returns:
    #         dict: metric_name -> numpy array of shape (N,)
    #     """
    #     imgs = images[:, 0, :, :]  # shape: (N, 32, 32)
    #
    #     # 1. L2 norm
    #     l2_norm = np.sqrt((imgs ** 2).sum(axis=(1, 2)))
    #
    #     # 2. Global pixel variance
    #     global_variance = imgs.var(axis=(1, 2))
    #
    #     # 3. Center patch variance
    #     center_patch = imgs[:, 8:24, 8:24]
    #     center_patch_variance = center_patch.var(axis=(1, 2))
    #
    #     # 4. Total variation with 4-neighborhood:
    #     # vertical and horizontal finite differences only
    #     vertical_tv = np.abs(imgs[:, 1:, :] - imgs[:, :-1, :]).mean(axis=(1, 2))
    #     horizontal_tv = np.abs(imgs[:, :, 1:] - imgs[:, :, :-1]).mean(axis=(1, 2))
    #     total_variation_4_neighbors = vertical_tv + horizontal_tv
    #
    #     return {
    #         "l2_norm": l2_norm,
    #         "global_variance": global_variance,
    #         "center_patch_variance": center_patch_variance,
    #         "total_variation_4_neighbors": total_variation_4_neighbors,
    #     }
    #
    #
    # def generate_final_samples(
    #     model,
    #     x_clean,
    #     device,
    #     stochastic,
    #     noise_level=1.0,
    #     batch_size=32,
    #     base_seed=1234,
    #     stochastic_noise_scale=1.0,
    # ):
    #     """
    #     Generates final samples for many images in small batches.
    #
    #     For fairness, deterministic and stochastic runs will receive the same degraded images
    #     and the same initial sampling noise, because we reset the seed before each call.
    #     """
    #     outputs = []
    #
    #     n = len(x_clean)
    #
    #     for start in range(0, n, batch_size):
    #         end = min(start + batch_size, n)
    #         x_batch = x_clean[start:end]
    #
    #         seed = base_seed + start
    #
    #         deg_batch = make_degraded_images(
    #             x_batch,
    #             noise_level=noise_level,
    #             seed=seed,
    #         )
    #
    #         # Make the initial random noise inside model.sample reproducible.
    #         torch.manual_seed(seed)
    #         if device == "cuda":
    #             torch.cuda.manual_seed_all(seed)
    #
    #         samples = model.sample(
    #             bsz=end - start,
    #             noise=noise_level,
    #             x0=deg_batch,
    #             device=device,
    #             stochastic=stochastic,
    #             stochastic_noise_scale=stochastic_noise_scale,
    #         )
    #
    #         # We only need the final output of the denoising process.
    #         final_samples = samples[-1].cpu().numpy()
    #         outputs.append(final_samples)
    #
    #     return np.concatenate(outputs, axis=0)
    #
    #
    # def summarize_metric_expectations(true_images, deterministic_images, stochastic_images):
    #     """
    #     Computes E[f(x)] for true, deterministic, and stochastic distributions.
    #     """
    #     true_metrics = compute_metric_values(true_images)
    #     det_metrics = compute_metric_values(deterministic_images)
    #     stoch_metrics = compute_metric_values(stochastic_images)
    #
    #     rows = []
    #
    #     for metric_name in true_metrics.keys():
    #         true_mean = float(true_metrics[metric_name].mean())
    #         det_mean = float(det_metrics[metric_name].mean())
    #         stoch_mean = float(stoch_metrics[metric_name].mean())
    #
    #         det_error = abs(det_mean - true_mean)
    #         stoch_error = abs(stoch_mean - true_mean)
    #
    #         if det_error < stoch_error:
    #             closer = "deterministic"
    #         elif stoch_error < det_error:
    #             closer = "stochastic"
    #         else:
    #             closer = "tie"
    #
    #         rows.append({
    #             "metric": metric_name,
    #             "true_E": true_mean,
    #             "deterministic_E": det_mean,
    #             "stochastic_E": stoch_mean,
    #             "det_abs_error": det_error,
    #             "stoch_abs_error": stoch_error,
    #             "closer_to_true": closer,
    #         })
    #
    #     return rows
    #
    #
    # # Number of images for distribution analysis.
    # # Start with 128. If runtime is reasonable, increase to 256 or 512.
    # N_EVAL = 512
    # ANALYSIS_BATCH_SIZE = 32
    # analysis_noise = 1.0
    #
    # x_eval = x[:N_EVAL].reshape(N_EVAL, 1, 32, 32).astype(np.float32)
    #
    # print("Running deterministic sampler for distribution analysis...")
    # deterministic_final = generate_final_samples(
    #     model=model,
    #     x_clean=x_eval,
    #     device=args.device,
    #     stochastic=False,
    #     noise_level=analysis_noise,
    #     batch_size=ANALYSIS_BATCH_SIZE,
    #     base_seed=1234,
    # )
    #
    # print("Running stochastic sampler for distribution analysis...")
    # stochastic_final = generate_final_samples(
    #     model=model,
    #     x_clean=x_eval,
    #     device=args.device,
    #     stochastic=True,
    #     noise_level=analysis_noise,
    #     batch_size=ANALYSIS_BATCH_SIZE,
    #     base_seed=1234,
    #     stochastic_noise_scale=1.0,
    # )
    #
    # rows = summarize_metric_expectations(
    #     true_images=x_eval,
    #     deterministic_images=deterministic_final,
    #     stochastic_images=stochastic_final,
    # )
    #
    # print("\nDistribution Analysis Results")
    # print("-" * 120)
    # header = (
    #     f"{'metric':35s} "
    #     f"{'true_E':>12s} "
    #     f"{'det_E':>12s} "
    #     f"{'stoch_E':>12s} "
    #     f"{'det_err':>12s} "
    #     f"{'stoch_err':>12s} "
    #     f"{'closer':>16s}"
    # )
    # print(header)
    # print("-" * 120)
    #
    # for row in rows:
    #     print(
    #         f"{row['metric']:35s} "
    #         f"{row['true_E']:12.6f} "
    #         f"{row['deterministic_E']:12.6f} "
    #         f"{row['stochastic_E']:12.6f} "
    #         f"{row['det_abs_error']:12.6f} "
    #         f"{row['stoch_abs_error']:12.6f} "
    #         f"{row['closer_to_true']:>16s}"
    #     )
    #
    # output_csv_path = "./examples/distribution_analysis.csv"
    #
    # with open(output_csv_path, "w", newline="") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=[
    #             "metric",
    #             "true_E",
    #             "deterministic_E",
    #             "stochastic_E",
    #             "det_abs_error",
    #             "stoch_abs_error",
    #             "closer_to_true",
    #         ],
    #     )
    #     writer.writeheader()
    #     writer.writerows(rows)
    #
    # print("-" * 120)
    # print(f"Saved distribution analysis to {output_csv_path}")
