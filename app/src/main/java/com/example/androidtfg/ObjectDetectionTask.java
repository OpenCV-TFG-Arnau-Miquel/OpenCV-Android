package com.example.androidtfg;

import android.os.AsyncTask;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

class ObjectDetectionTask extends AsyncTask<Mat, Object, Object> {

    float confThreshold = 0.5f;
    float nmsThresh = 0.2f;

    private Net net;
    private final MainActivity listener;

    public ObjectDetectionTask(Net net, MainActivity listener) {
        this.net = net;
        this.listener = listener;
    }

    @Override
    protected Object doInBackground(Mat... frames) {
        Mat frame = frames[0];

        // Get all the bounding boxes from the network
        List<Mat> result = detect(frame);

        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> boxes = new ArrayList<>();

        processDetections(frame, result, clsIds, confs, boxes);

        filterDetections(clsIds, confs, boxes);

        return  null;
    }

    private List<Mat> detect(Mat frame) {


        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Generate a 4-dimensional Binary Object from image, after mean subtraction, normalizing, and channel swapping.
        // 4-dimensions: num_images, num_channels (colors), width, height.
        Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),false,false);

        // Pass the binary image to the network
        net.setInput(imageBlob);

        List<Mat> result = new ArrayList<>();

        // Get the names of the layers in the network
        List<String> outBlobNames = getOutputNames(net);

        // Runs forward pass to get the output result of the passed output layers
        // The result is a set of 4-dimensional Binary Objects, corresponding to each layer output
        // in this case we pass 3 layers, so 3 results.
        net.forward(result, outBlobNames);
        return result;
    }

    private void processDetections(Mat frame, List<Mat> result, List<Integer> clsIds, List<Float> confs, List<Rect> boxes) {


        // Scan all layers output
        for (int i = 0; i < result.size(); ++i) {

            Mat level = result.get(i); // a level is a layer output

            // See all detections for each layer
            for (int j = 0; j < level.rows(); ++j) {

                Mat detection = level.row(j); // row in a level is a detection

                Mat scores = detection.colRange(5, level.cols()); // information about class predictions
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores); // Get values and locations of possible classes
                float confidence = (float) mm.maxVal; // max probability of class
                Point classIdPoint = mm.maxLoc; // position of the class in the set

                // Only keep high confidence bounding boxes
                if (confidence > confThreshold) {
                    // get data of the bounding box in relation to the main frame
                    // center of the box
                    // width and height
                    // values are divisors (0.5, 0.65...)
                    int centerX = (int) (detection.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (detection.get(0, 1)[0] * frame.rows());
                    int width = (int) (detection.get(0, 2)[0] * frame.cols());
                    int height = (int) (detection.get(0, 3)[0] * frame.rows());

                    // get start points of the bounding box in the frame
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);   // x is the position needed to get the class name form the array
                    confs.add(confidence);  // confidence will be in same place in array than the corresponding class in the array above

                    boxes.add(new Rect(left, top, width, height));  // save this box: start position in frame + sizes
                }
            }
        }
    }

    private void filterDetections(List<Integer> clsIds, List<Float> confs, List<Rect> boxes) {


        int ArrayLength = confs.size();
        if (ArrayLength >= 1) {

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect[] boxesArray = boxes.toArray(new Rect[0]);

            MatOfRect boxesMat = new MatOfRect(boxesArray);

            MatOfInt indices = new MatOfInt();  // it will save the new mapping of the previous detections when filtered

            // Remove redundant overlapping boxes that have lower confidences - filter boxes
            Dnn.NMSBoxes(boxesMat, confidences, confThreshold, nmsThresh, indices);

            int[] ind = indices.toArray();
            ArrayList<Detection> detections = new ArrayList<>();
            for (int i = 0; i < ind.length; ++i) {  // save the remaining boxes

                int idx = ind[i];
                Rect box = boxesArray[idx];

                int id = clsIds.get(idx);

                float conf = confs.get(idx);

                int intConf = (int) (conf * 100); // get value in hundred per cent units

                Point textPosition = box.tl().clone();
                textPosition.set( new double[]{textPosition.x, textPosition.y - 10}); // move text a little bit upper

                detections.add(new Detection(box, id, intConf, textPosition)); // save to the array that will be passed to mainActivity

            }

            listener.setNewDetections(detections);  //pass the array of detections to the mainActivity

        }

    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        // unfold and create R-CNN layers from the loaded YOLO model
        for (Integer item : outLayers) {
            names.add(layersNames.get(item - 1));
        }
        return names;
    }

    @Override
    protected void onCancelled(Object o) {
        super.onCancelled(o);
    }


}
