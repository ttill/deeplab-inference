# MT inference

Run inference on binary (2 classes only) Deeplab model cropping large images in a sliding window approach.

## Prerequisites

You will need the following things properly installed on your computer:
* [Git](https://git-scm.com/)
* Python >= 3.6
* pipenv or docker

## Installation / Setup

* `git clone git@github.com:ttill/deeplab-inference.git`
* `cd deeplab-inference`

### Locally

* `pipenv install`

### Using docker

* `docker build -t deeplab-inference .`


## Running

```
Usage: app.py [OPTIONS]

  Run inference on binary (2 classes only) Deeplab model stored in frozen
  graph on image. Image has to be in RGB format. It will be cropped in a
  sliding window approach (see option `crop_size`).

Options:
  --frozen-graph FILE             Path to frozen graph of the model to use in
                                  protobuf format.  [required]

  --input PATH                    Path to input image to infer on. In case a
                                  directory is passed inference will be run on
                                  every contained.  [required]

  --crop-size INTEGER             In case input is larger in terms of width or
                                  height, it will be cropped into patches of
                                  `crop_size` with overlaps of 50 pixels will
                                  be passed into the model. Predicitons will
                                  be stitched together considering the maximum
                                  value in the overlapping regions.

  --segmentation PATH             Where to put a PNG image of the segmentation
                                  map. Filename or folder (uses basename of
                                  input).

  --probability PATH              Where to put a PNG pseudocolor image of the
                                  probability map. Filename or folder (uses
                                  basename of input).

  --visualize-progress / --no-visualize-progress
                                  Show figure containing input, probability
                                  map (pseudocolors), segmentation overlay for
                                  every cropped patch (halts inference).

  --visualize-result / --no-visualize-result
                                  Show figure containing input, probability
                                  map (pseudocolors), segmentation overlay for
                                  every input image (halts inference).

  --ground-truth PATH             Path to image (filename or folder where png
                                  with same basename as input is located) with
                                  ground truth as 8-bit grayscale png (class
                                  0: `0`, class 1: `255`). When supplied
                                  multiple metrics will be calculated
                                  (Accuracy, MIoU, Matthews Correlation
                                  Coefficient, …)

  --split-file FILE               Use this to filter input folder. Path to
                                  text file with one basename per line (pascal
                                  voc format).

  --difference PATH               Where to put a PNG image containing a
                                  difference map: Black: True negative (class
                                  0), White: True positive (class 1), Red:
                                  False negative, Blue: False positive.
                                  Filename or folder (uses basename of input).

  --install-completion            Install completion for the current shell.
  --show-completion               Show completion for the current shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```

### Locally

* `pipenv run python app.py --frozen-graph path/to/frozen_inference_graph.pb --input path/to/images/ --segmentation output/folder/`


### Using docker

* `docker run -it -v /path/to/data/:/data/ deeplab-inference bash`
* `python app.py --frozen-graph /data/frozen_inference_graph.pb --input /data/images/ --segmentation /data/segmentation/`

## License

This project is licensed under the [MIT License](LICENSE).
