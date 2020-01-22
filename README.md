# RAMP starting kit on the Titanic dataset

Authors: Alexandre Gramfort & Balazs Kegl

[![Build Status](https://travis-ci.org/ramp-kits/titanic.svg?branch=master)](https://travis-ci.org/ramp-kits/titanic)

Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

to test the starting kit submission (`submissions/starting_kit`).

To test a different submission use:

```
ramp_test_submission --submission=<folder>
```

where `<folder>` is the name of the folder within `submissions` that your
submission is saved under. For example to test `submissions/random_forest_20_5`
use:

```
ramp_test_submission --submission=random_forest_20_5
```

Get started on this RAMP with the [dedicated notebook](titanic_starting_kit.ipynb).
