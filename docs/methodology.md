# Methodology

## Units and sign convention

The analysis uses volts and mA/cm². Power density is therefore expressed in mW/cm², and the local-slope resistance estimates are expressed in Ω·cm².

The input current sign is normalized so that generated current is positive near 0 V. The median of the five measurements closest to 0 V determines the sign factor.

## Extracted metrics

### Short-circuit current density

Jsc is linearly interpolated at 0 V. When a sweep does not bracket 0 V, a local linear extrapolation is used and the result includes a warning.

### Open-circuit voltage

Voc is the first positive-voltage zero-current crossing, calculated by linear interpolation between adjacent measurements.

### Maximum power point

Pmax is the largest measured value of `V × J` in the power-generating quadrant. Vmp and Jmp are the corresponding measured coordinates.

### Fill factor and efficiency

```text
FF = Pmax / (Voc × Jsc)
PCE (%) = 100 × Pmax / incident power density
```

### Series and shunt resistance

Rs and Rsh are estimated from local linear fits near Voc and 0 V:

```text
Rs  = -1000 / (dJ/dV near Voc)
Rsh = -1000 / (dJ/dV near 0 V)
```

These values are useful for screening. They are not substitutes for diode-model fitting.

## Anomaly checks

### Hysteresis

Forward and reverse scans are interpolated on a common grid from 0 V to the lower extracted Voc. The reported index is the larger of:

- relative PCE difference
- normalized area between the two curves

### S-kink

The score examines the normalized curve between 0.15 Voc and 0.95 Voc. Savitzky-Golay smoothing is applied before calculating slope and curvature. Small derivative excursions are ignored to reduce sensitivity to measurement noise. The score is a screening heuristic and requires calibration on reviewed curves.

### Shunting and poor fill factor

Shunting is reported when Rsh falls below its configured threshold. Poor fill factor is reported when FF falls below its configured threshold.

## Random Forest model

The optional classifier applies median imputation and a balanced Random Forest. The model bundle stores its feature schema and format version so incompatible files fail clearly.

The anomaly probability equals one minus the combined probability of configured normal labels. The highest-probability anomaly class is reported when that probability exceeds the configured threshold.

## Qualification guidance

Before using the pipeline for release decisions:

1. Verify column mapping and current sign against reference files.
2. Calibrate pass/fail thresholds by device technology and process stage.
3. Compare Rs and Rsh with an accepted fitting method.
4. Evaluate anomaly labels on an independent, representative test set.
5. Version the configuration, model, and training data with every report batch.

