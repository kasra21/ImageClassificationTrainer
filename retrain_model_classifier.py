import sys
import tensorflow as tf
import numpy as np

#the argument for the image to analyze
image_path = sys.argv[1]
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# Loads labels
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./output.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph to get predications
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-len(predictions):][::-1]

    for node_id in top_k:
      label = label_lines[node_id]
      accuracy = predictions[node_id]
      print('%s (accuracy = %.5f)' % (label, accuracy))