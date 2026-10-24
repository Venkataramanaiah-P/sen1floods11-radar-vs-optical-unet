# Does adding radar help flood mapping? U-Nets on Sentinel-1 + Sentinel-2 (Sen1Floods11)

I trained flood-mapping U-Nets on three kinds of input (optical only, radar only, both stacked) and compared them with two simple rules that use no learning. Every number below comes from the same notebook, scored on the same pixels.

![Test IoU for the simple rules and the three U-Nets](results_chart.png)

## TL;DR

- **Learning beat the simple rules for both sensors:** optical +0.057 over NDWI, radar +0.130 over a VH threshold.
- **Radar alone is stable but well behind optical:** about 0.65 against 0.83 test IoU.
- **Adding radar to optical gave no gain in this experiment**, and the fused model was less consistent from run to run. I did not find evidence that radar helped here. I am not claiming it hurts, because three runs per model is a small sample and the ranges overlap.

## Results (test IoU, 90 test chips)

| Method | Mean test IoU | Lowest | Highest | Runs |
|---|---|---|---|---|
| NDWI rule (optical), cut-off 0.0 | 0.772 | n/a | n/a | 1 |
| VH rule (radar), cut-off -24 dB | 0.523 | n/a | n/a | 1 |
| **U-Net, optical only (13 layers)** | **0.829** | 0.821 | 0.833 | 3 |
| U-Net, radar only (2 layers) | 0.653 | 0.648 | 0.657 | 3 |
| U-Net, optical + radar stacked (15 layers) | 0.812 | 0.792 | 0.830 | 3 |

Per-run scores, best epochs and validation scores are in `results.csv`.

The two rules were tuned on the training chips only (NDWI: train 0.651, valid 0.672; VH: train 0.464, valid 0.488), then scored once on the test chips.

## What I did

**Data.** Sen1Floods11 (public bucket `gs://sen1floods11`): 446 hand-labelled chips from 11 flood events. Each chip is 512 x 512 pixels at 10 m, with Sentinel-1 (VV and VH, in dB) and Sentinel-2 (13 bands) over the same area. Labels are 1 = water, 0 = not water, -1 = not labelled.

**Splits.** I used the train / valid / test lists published with the dataset: 252 / 89 / 90 chips. The 15 Bolivia chips are in none of the three lists, so they were not used.

**Usable pixels.** A pixel counts only if a person labelled it, the radar has a reading there, and the optical image is not empty there. Every method, rules and U-Nets alike, is scored on exactly those pixels.

**Score.** Intersection over union (IoU) of the predicted water and the labelled water, pooled over all test pixels. Water is only about 10.6% of labelled pixels, so plain accuracy would be misleading.

**Simple rules.** NDWI = (Green - NIR) / (Green + NIR), water if above a cut-off. VH radar, water if below a cut-off in dB. I tried a list of cut-offs, picked the best on the training chips, and checked it on validation and test.

**Model.** U-Net with a ResNet34 encoder (ImageNet-pretrained) from `segmentation-models-pytorch`. Same settings for all three input types: 30 epochs, batch 8, random 256 x 256 crops with flips, Adam at 1e-3, loss = binary cross-entropy + Dice, counted only on usable pixels. The best epoch is chosen on the validation chips. The test chips are scored once per run. Each run takes roughly 5-6 minutes on a Colab GPU.

**Repeats.** Three seeds per input type.

## Fairness rules built into the notebook

1. Cut-offs and the best epoch are chosen on train and validation chips only.
2. Test chips are scored once per run.
3. Every method is scored on the same usable pixels.
4. Only the input layers change between the three U-Net experiments. Nothing else is tuned per input type.

## What the runs showed

- **Run-to-run noise is real.** Validation IoU swings a lot between epochs (for optical it typically sits around 0.74 and peaks near 0.80). The best-epoch score is the top of a noisy line, so the test score is the number to trust.
- **Same seed did not reproduce exactly on a GPU.** An earlier set of optical runs with the same code gave test scores of 0.799 to 0.827, against 0.821 to 0.833 in the final set. I treat differences below about 0.03 as noise.
- **Data notes:** about 10% of chips (46 of 446) have blank radar readings, and about 94% of the blank radar pixels are also unlabelled, so dropping them costs very few labelled pixels.

## Limitations


1. **Label origin.** The labels were made with help from Sentinel-2 imagery, so optical inputs may have an advantage. The result may not hold for labels made another way.
2. **cloud simulation is in Part 2.** Radar's main advantage is seeing through cloud, and I did not simulate cloud-covered optical images.
3. **One fusion design.** I only tried early fusion (stacking all layers). Other designs, such as separate encoders per sensor, may behave differently.
4. **Same events in train and test.** The test chips come from the same flood events as the training chips, which is easier than predicting a flood in a new region.
5. **Small sample.** Three runs per input type shows whether the ranges overlap, not a precise difference.

## What I would test next

- Hide parts of the optical image to simulate cloud, and compare optical-only with fused.
- Hold out a whole flood event for testing.
- Try a fusion design with separate radar and optical encoders.

## Reproduce

1. Open `F1_flood_SAR_optical_complete.ipynb` in Google Colab and choose Runtime -> Change runtime type -> a GPU.
2. Run Part 0 to Part 5 in order. Part 0 downloads the data from the public bucket `gs://sen1floods11`.
3. For one input type at a time, set `MODE` to `"s2"`, `"s1"` or `"both"` in Part 1, then run Parts 6A and 6B. For everything at once, set `RUN_ALL = True` and run only Part 6C (about 50 minutes for 3 seeds x 3 inputs).
4. Run Part 7 for the summary table.

## Files

| File | What it is |
|---|---|
| `F1_flood_SAR_optical_complete.ipynb` | The whole project, with a remark beside every code line |
| `results.csv` | One line per run: input type, seed, encoder, weights, epochs, best epoch, validation IoU, test IoU |
| `results_chart.png` | The chart above |

## Data source and citation

Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.

## Author

Venkataramanaiah Poliboyina. 25 years of infrastructure survey experience, now working in GeoAI and remote sensing.
[GitHub](https://github.com/Venkataramanaiah-P) | [LinkedIn](https://www.linkedin.com/in/venkataramanaiahpoliboyina)


## Part 2: Where does radar start to help? (cloud-mask experiment)

Part 1 found no clear benefit from adding radar on clear-sky test chips. Radar (Sentinel-1) is useful because clouds cannot hide it, so this part asks a different question: **how much of the optical image has to be missing before radar starts to matter?**

This test was suggested by Adam Ruf.

### Method

- Same U-Net (ResNet34 encoder) and same Sen1Floods11 split as Part 1.
- Clouds are **simulated**: a smooth random "cloudiness" map is made for each chip from a fixed seed, and the optical layers (the first 13 input channels) are set to zero where it is cloudy. **Radar layers are never touched.**
- The same cloud map is used for the optical-only and the fused model on each chip and level, so the comparison is fair.
- Both models are trained with random clouds (20% of training chips are clear sky, the rest have a random share of the optical image visible) and use the same settings as Part 1 (30 epochs, 3 seeds: 0, 1, 2). The best epoch is picked on a mix of 100%, 50% and 25% visible.
- Test levels: 100%, 75%, 50%, 25%, 10% and 0% of the optical image visible.
- The dashed line in the chart is the radar-only U-Net from Part 1 (test IoU 0.653), which clouds cannot affect.

### Results (test IoU)

| Optical visible | Optical only | Fused (optical + radar) | Radar gain |
|---|---|---|---|
| 100% | 0.805 | 0.808 | +0.003 |
| 75% | 0.764 | 0.780 | +0.016 |
| 50% | 0.718 | 0.748 | +0.030 |
| 25% | 0.648 | 0.709 | +0.061 |
| 10% | 0.527 | 0.677 | +0.149 |
| 0% | 0.000 | 0.561 | +0.561 |

![Where does radar start to help?](cloud_plot.png)

### What this shows

1. **On a clear day radar adds almost nothing** (+0.003 IoU). This is smaller than the difference between seeds (optical-only ranges 0.790 to 0.814, fused 0.803 to 0.813), so it is noise, which is consistent with Part 1.
2. **The more cloud, the more radar helps.** The gain grows from +0.03 at half cloud to +0.15 when only 10% of the optical image is visible. From 75% visible downwards the fused model's worst seed beats the optical-only model's best seed.
3. **Below about 25% visible, the optical-only model is worse than the radar-only model** (0.65). Past that point, an optical-only flood map is worse than one made from radar alone.
4. **A weakness I did not expect:** when the optical image is completely hidden (0%), the fused model scores 0.561, which is **lower than the radar-only model (0.653)**. The three seeds give 0.503, 0.565 and 0.616, so all are below it, but the spread is large. This fused model does not fully fall back to radar. Possible reasons (not tested) are that it only saw a small share of fully hidden chips in training, and that it must also cope with missing optical channels.

### Limitations

- Clouds are simulated with smooth noise. Real clouds also leave haze and shadows that change the optical values instead of zeroing them.
- Only the optical layers are hidden. Real cloud cover also changes the scenes that were chosen for the dataset.
- One dataset (Sen1Floods11), one architecture and one fusion design (channel stacking).
- The clear-sky numbers here (optical 0.805, fused 0.808) are lower than Part 1 (0.829 and 0.812) because these models were trained with clouds in every epoch. Compare numbers only within the same part.
- Shaded bands in the chart show the spread across the 3 seeds, not a formal confidence interval. Three seeds is a small sample.

### Files added

- `F1_flood_cloud_experiment.ipynb`: the notebook for this part (Colab T4).
- `cloud_results.csv`: every test result (mode, seed, visibility level, IoU).
- `cloud_plot.png`: the chart above.

### Reproduce

Open `F1_flood_cloud_experiment.ipynb` in Google Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the cells in order. The last cells train both models with clouds, test them at each visibility level, and draw the plot.
