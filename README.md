# aind-registration-evaluation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

This package evaluates a transformation applied in two large-scale images. After taking both images to the same coordinate system, we sample points in the intersection area that we later use to compute the metric.
The input vales are:
- Image 1: Path where image 1 is located. It could be 2D or 3D.
- Image 2: Path where image 2 is located. It could be 2D or 3D.
- Transform matrix: List of list that have the transformation matrix.
- Data type: Scale of the data. Small refers to data that can fit in memory and large for data that can't.
- Metric: Acronym of the metric that will be used for evaluation.
- Window size: For every point, we take an are equal to 2 * window_size + 1 which creates a square or cube for the two images in the same location of the intersection area.
- Sampling info: A dictionary that contains the number of points that will be sampled in the intersection area as well as the sampling type. At this moment, we sample points randomly or in a grid.

![example](https://github.com/AllenNeuralDynamics/aind-registration_evaluation/blob/main/images/example_evaluation.png?raw=true)

## Data type
Due to the large-scale data the Allen Institute for Neural Dynamics is generating, we are lazily reading image data and computing the metrics in chunks when the data is large (this means the data can't fit in standard computers memory). In these cases, we recommend setting the data type to `large` and `small` for images that could fit in memory. Selecting this depends on your use case, resources and nature of your data.

![DataType](https://github.com/AllenNeuralDynamics/aind-registration_evaluation/blob/main/images/diagram_evaluation.png?raw=true)

## Metrics
We have the most common computer vision metrics to evaluate images. Here, we include the following metrics:
| Metric         | Acronym     | Data scale |
|--------------------|---------|------------|
| Mean Squared Error | ssd     | :white_check_mark: Large :white_check_mark: Small     |
| Structural Similarity Index | ssim     | :white_check_mark: Large :white_check_mark: Small     |
| Mean Absolute Error | mae     | :white_check_mark: Large :white_check_mark: Small     |
| R2 Score | r2     | :white_check_mark: Large :white_check_mark: Small     |
| Max Error | max_err     | :white_check_mark: Large :white_check_mark: Small     |
| Normalized Cross Correlation | ncc     | :white_check_mark: Large :white_check_mark: Small     |
| Mutual Information | mi     | :white_check_mark: Large :white_check_mark: Small     |
| Normalized Mutual Information | nmi     | :white_check_mark: Large :white_check_mark: Small     |
| Information Theoretic Similarity | issm     | :white_check_mark: Small     |
| Peak Signal to Noise Ratio | psnr     | :white_check_mark: Large :white_check_mark: Small     |
| Feature Similarity Index Metric  | fsim     | :white_check_mark: Small     |
| Keypoint Similarity Alignment Metric  | ksam     | :white_check_mark: Small     |

## Transform matrix
The matrix is in homogeneous coordinates. Therefore, for a 2D image the matrix will be:

$$\begin{bmatrix}
{y_{11}}&{y_{12}}&{\cdots}&{y_{1n}}\\
{x_{21}}&{x_{22}}&{\cdots}&{x_{2n}}\\
{w_{m1}}&{w_{m2}}&{\cdots}&{w_{mn}}\\
\end{bmatrix}$$ 

and for 3D:

$$\begin{bmatrix}
{z_{21}}&{x_{22}}&{\cdots}&{z_{2n}}\\
{y_{11}}&{y_{12}}&{\cdots}&{y_{1n}}\\
{x_{21}}&{x_{22}}&{\cdots}&{x_{2n}}\\
{w_{m1}}&{w_{m2}}&{\cdots}&{w_{mn}}\\
\end{bmatrix}$$

## Window size
This refers to how big the area around each sampled point will be. For example, in a 2D image the window size area for a given point will be:

![PointWindowSize](https://github.com/AllenNeuralDynamics/aind-registration_evaluation/blob/main/images/point_window_size.png?raw=true)

The same applies for a 3D sampled point.

## Sampling options
The sampling options we have at the moment are:
- Sampling type: This could be a grid of points located in the intersection area or points spread randomly. `Options = ["grid", "random"]`
- Number of points: The approximate number of points that will be sampled in the image.

## Available image formats
The available image formats at the moment are:
- Zarr
- Tiff

## Keypoint Similarity Alignment Metric
The keypoint similarity alignment metric is a metric that is currently being developed at the Allen Institute for Neural Dynamics as part of the image stitching evaluation protocol for large-scale image datasets. Please, refer to `/scripts/run_misalignment.py` example. The current options are `"energy"` and `"maxima"` methods. Please, read the documentation to learn more about these options.

These are some metric examples:
- Image stitching alignment aberration
![multichannel_misalign_1](https://github.com/AllenNeuralDynamics/aind-registration_evaluation/blob/main/images/keypoint_metric_examples/multichannel_example_2.png?raw=true)

- Improved image stitching alignment:
![multichannel_misalign_2](https://github.com/AllenNeuralDynamics/aind-registration_evaluation/blob/main/images/keypoint_metric_examples/multichannel_example.png?raw=true)

## Note
> If you want to visualize points in the images, you could activate the visualize flag the package accepts. However, make sure these images are able to fix in memory.

## Run Full Report
To run the full report find the full_report.py class in scripts and move it in the source folder. You can adjust what metrics you are interested in by changing the parameters of the ImageAnalysis instantiation. 

By running run_to_excel, you will create an excel file, displaying all the results of the analysis. These include MI and NMI for the following evaluation approaches:
- Run the analysis on the full image and output the results unnormalized and normalized
- Run the analysis on Regions of interest patches. These ROIs are defined as closely located, labeled regions across the two images, based on the inputted masks.
- Run the analysis on the full matching masks. Matching masks are created by leaving the labeled matching ROIs in the image and nulling everything else.
- Run the analysis on the ROI patches of the matching masks.

- Additionally, you will receive the number of ROIs (e.g. cells in each image) and the number of matches across the two images based on location
- Lastly, on another sheet (sheetname linked to a certain run), you will find the fraction of matching cells across the two images out of the maximum number of cells.

Example Report:
![image](https://github.com/user-attachments/assets/6ccf2842-65d4-4d6e-81c4-75b0f6bb2d7a)
![image](https://github.com/user-attachments/assets/c18db2ce-536d-40d0-bd3f-88e1ee324993)






## Installation
To use the software, in the root directory, run
```
pip install -e .
```

To develop the code, run
```
pip install -e .[dev]
```

## To Run
Run with the following:

```
python scripts/evaluation_example.py
```

The example_input dict at the top of the file gives an example of the inputs. If "datatype" is set to dummy, then it will create dummy data and run it on that. If it's set to "large" then it will try to read the zarr files.

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
