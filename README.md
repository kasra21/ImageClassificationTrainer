# Image Classification Trainer

This project is here to generate a graph (only the final layer of the graph) for the tensorflow and retrain the tensorflow neural network. After retraining a set of labels (labels.txt) and the graph (output.pb) will be generated. These two files can then be used to classify an image to the labels generated.

### How do I get set up?

* **Clone The project**: 

	 git clone https://kasra21@bitbucket.org/kasra21/imageclassificationtrainer.git

* **Details**:

* **Build/Usage**:

To retrain you may run:

	python retrain.py --model_dir ./inception --image_dir <path_to_the_images> --output_graph ./output.pb --output_labels ./labels.txt --how_many_training_steps <number_of_training>

the higher the `<number_of_training>` is the better chance and accuracy we may get.
To text the result and classifying an image, you may run the following:

	Python retrain_model_classifier.py <path_to_the_image_to_classify>