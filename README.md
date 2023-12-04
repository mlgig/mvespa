# Machine Vision-Enabled Sports Performance Analysis

This code capsule reproduces the results presented in the manuscript with the title above. View the results in [`results/results.ipynb`](/results/results.ipynb).

**_Goal:_** This study investigates the feasibility of monocular 2D markerless motion capture (MMC) using a single smartphone to measure jump height, velocity, flight time, contact time, and range of motion (ROM) during motor tasks. **_Methods:_** Sixteen healthy adults performed three repetitions of selected tests while their body movements were recorded using force plates, optical motion capture (OMC), and a smartphone camera. MMC was then performed on the smartphone videos using OpenPose v1.7.0. **_Results:_** MMC demonstrated excellent agreement with ground truth for jump height and velocity measurements. However, MMC's performance varied from poor to moderate for flight time, contact time, ROM, and angular velocity measurements. **_Conclusions:_** These findings suggest that monocular 2D MMC may be a viable alternative to OMC or force plates for assessing sports performance during jumps and velocity-based tests. Additionally, MMC could provide valuable visual feedback for flight time, contact time, ROM, and angular velocity measurements.
**_Keywords:_** `Markerless motion capture`, `Optical motion capture`, `Sports performance analysis`, `Smartphone motion capture`

## How To Use

To reproduce the results

- Download this repository and unzip.
- In the root directory, run `pip install -r requirements.txt`.
- Run `run.bat` or `run.sh`. This will run the code in `results.ipynb` and may take between 60-120 seconds. You can then see the result in [`results/results.ipynb`](/results/results.ipynb).
  -- _Alternatively, you may also run `jupyter notebook` and manually run the `results.ipynb` notebook cell-by-cell._
