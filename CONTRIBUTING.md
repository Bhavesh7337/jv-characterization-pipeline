# Contributing

Use a focused branch for each change and include a test when behavior changes.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
```

Keep input parsing, metric extraction, classification, and reporting concerns in their existing modules. Avoid changing default acceptance thresholds without documenting the reason and the validation data used.

