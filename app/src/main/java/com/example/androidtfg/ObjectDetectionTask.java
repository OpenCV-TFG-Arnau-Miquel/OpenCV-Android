package com.example.androidtfg;

import android.os.AsyncTask;
import android.util.Log;

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
    private static final String TAG = "ObjectDetectionTask";

    float confThreshold = 0.3f;
    float nmsThresh = 0.2f;

    private Net net;
    private final MainActivity listener;

    private ArrayList<String> cocoNames;

    public ObjectDetectionTask(Net net, ArrayList<String> cocoNames, MainActivity listener) {
        this.cocoNames = cocoNames;
        this.net = net;
        this.listener = listener;
    }

    @Override
    protected Object doInBackground(Mat... frames) {
        Log.d(TAG, "START DETECTION - BACKGROUND");
        Mat frame = frames[0];

        // Get all the bounding boxes from the network
        List<Mat> result = generateResults(frame);

        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> boxes = new ArrayList<>();

        detect(frame, result, clsIds, confs, boxes);

        addDetections(clsIds, confs, boxes);

        return  null;
    }

    private List<Mat> generateResults(Mat frame) {
        Log.d(TAG, "GENERATE RESULTS - BACKGROUND");


        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Generate Binary Object
        Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);

        // Pass the binary image to the network
        net.setInput(imageBlob);

        List<Mat> result = new ArrayList<>();

        // Get the names of the layers in the network
        List<String> outBlobNames = getOutputNames(net);

        // Get the output result of the passed output layers
        net.forward(result, outBlobNames);
        return result;
    }

    private void detect(Mat frame, List<Mat> result, List<Integer> clsIds, List<Float> confs, List<Rect> boxes) {
        Log.d(TAG, "DETECT - BACKGROUND");


        // Scan all bounding boxes
        for (int i = 0; i < result.size(); ++i) {

            Mat level = result.get(i);

            // All detections for current box
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());

                // Get value and location of maximum value
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);

                float confidence = (float) mm.maxVal;

                Point classIdPoint = mm.maxLoc;

                // Only keep high confidence bounding boxes
                if (confidence > confThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);
                    confs.add(confidence);

                    boxes.add(new Rect(left, top, width, height));
                }
            }
        }
    }

    private void addDetections(List<Integer> clsIds, List<Float> confs, List<Rect> boxes) {
        Log.d(TAG, "ADD DETECTIONS - BACKGROUND");


        int ArrayLength = confs.size();
        if (ArrayLength >= 1) {

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect[] boxesArray = boxes.toArray(new Rect[0]);

            MatOfRect boxesMat = new MatOfRect(boxesArray);

            MatOfInt indices = new MatOfInt();

            // Remove redundant overlapping boxes that have lower confidences
            Dnn.NMSBoxes(boxesMat, confidences, confThreshold, nmsThresh, indices);

            int[] ind = indices.toArray();
            ArrayList<Detection> detections = new ArrayList<>();
            for (int i = 0; i < ind.length; ++i) {

                int idx = ind[i];
                Rect box = boxesArray[idx];

                int id = clsIds.get(idx);

                float conf = confs.get(idx);

                int intConf = (int) (conf * 100);

                Point tl = box.tl().clone();
                tl.set( new double[]{tl.x, tl.y - 10});

                Log.d(TAG, "Object detected -> " + cocoNames.get(id).toUpperCase() + " " + intConf + "%");
                detections.add(new Detection(box, id, intConf, tl));

            }

            listener.setNewDetections(detections);

        }

    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        //unfold and create R-CNN layers from the loaded YOLO model//
        for (Integer item : outLayers) {
            names.add(layersNames.get(item - 1));
        }
        return names;
    }


}
