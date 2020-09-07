package com.example.androidtfg;

import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    float confThreshold = 0.3f;
    float nmsThresh = 0.2f;

    private static final String TAG = "MainActivity";
    CameraBridgeViewBase cameraBridgeViewBase;
    Net net;
    String yoloCfg;
    String yoloWeights;
    private StorageReference mStorageRef;
    private ArrayList<String> cocoNames;
    private boolean netInitialized = false;

    public static final int REQUEST_ID_MULTIPLE_PERMISSIONS= 3;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        mStorageRef = FirebaseStorage.getInstance().getReference();

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There's a problem", Toast.LENGTH_SHORT).show();
        }

    }

    @Override
    protected void onStart() {
        Log.d(TAG, "onStart");

        super.onStart();

        requestAppPermissions();
    }

    @Override
    protected void onResume() {
        Log.d(TAG, "onResume");
        super.onResume();

    }

    @Override
    protected void onPause() {
        Log.d(TAG, "onPause");

        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onStop() {
        Log.d(TAG, "onStop");

        super.onStop();
    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy");

        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.d(TAG, "FRAME PROCESSING");

        Mat frame = inputFrame.rgba();

        formatFrame(frame);

        if (netInitialized) {

            // Get all the bounding boxes from the network
            List<Mat> result = generateResults(frame);

            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> boxes = new ArrayList<>();

            detect(frame, result, clsIds, confs, boxes);

            addDetections(frame, clsIds, confs, boxes);
        }

        return frame;
    }

    private void formatFrame(Mat frame) {
        Core.rotate(frame, frame, 360);
        Core.flip(frame, frame, 0);
        Core.flip(frame, frame, 1);
    }

    private List<Mat> generateResults(Mat frame) {
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

    private void addDetections(Mat frame, List<Integer> clsIds, List<Float> confs, List<Rect> boxes) {
        int ArrayLength = confs.size();
        if (ArrayLength >= 1) {

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect[] boxesArray = boxes.toArray(new Rect[0]);

            MatOfRect boxesMat = new MatOfRect(boxesArray);

            MatOfInt indices = new MatOfInt();

            // Remove redundant overlapping boxes that have lower confidences
            Dnn.NMSBoxes(boxesMat, confidences, confThreshold, nmsThresh, indices);

            int[] ind = indices.toArray();
            for (int i = 0; i < ind.length; ++i) {

                int idx = ind[i];
                Rect box = boxesArray[idx];

                int id = clsIds.get(idx);

                float conf = confs.get(idx);

                int intConf = (int) (conf * 100);

                Point tl = box.tl().clone();
                tl.set( new double[]{tl.x, tl.y - 10});

                drawDetections(frame, box, id, intConf, tl);

            }
        }
    }

    private void drawDetections(Mat frame, Rect box, int id, int intConf, Point tl) {
        Imgproc.putText(frame, cocoNames.get(id).toUpperCase() + " " + intConf + "%", tl, Core.FONT_HERSHEY_COMPLEX, 0.75, new Scalar(255, 255, 0), 1);
        Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "onCameraViewStarted");

    }

    @Override
    public void onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped");

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_ID_MULTIPLE_PERMISSIONS: {

                Map<String, Integer> perms = new HashMap<>();
                perms.put(CAMERA, PackageManager.PERMISSION_GRANTED);
                perms.put(WRITE_EXTERNAL_STORAGE, PackageManager.PERMISSION_GRANTED);
                perms.put(READ_EXTERNAL_STORAGE, PackageManager.PERMISSION_GRANTED);
                if (grantResults.length > 0) {
                    for (int i = 0; i < permissions.length; i++) {
                        perms.put(permissions[i], grantResults[i]);
                    }

                    if (perms.get(WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                            && perms.get(CAMERA) == PackageManager.PERMISSION_GRANTED
                            && perms.get(READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                        cameraBridgeViewBase.enableView();
                        initObjectDetection();
                    } else {
                        if (ActivityCompat.shouldShowRequestPermissionRationale(this, WRITE_EXTERNAL_STORAGE) || ActivityCompat.shouldShowRequestPermissionRationale(this, CAMERA) || ActivityCompat.shouldShowRequestPermissionRationale(this, READ_EXTERNAL_STORAGE)) {
                            showDialogOK("You must accept Camera and Storage Permission as they are required for this app",
                                    (dialog, which) -> {
                                        switch (which) {
                                            case DialogInterface.BUTTON_POSITIVE:
                                                requestAppPermissions();
                                                break;
                                            case DialogInterface.BUTTON_NEGATIVE:
                                                finish();
                                                break;
                                        }
                                    });
                        } else {
                            Toast.makeText(this, "You should enable permissions in your device settings", Toast.LENGTH_LONG).show();
                        }
                    }
                }
            }
        }
    }

    private void requestAppPermissions() {
        List<String> permissions =  new ArrayList<>();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(CAMERA);
            }
            if (checkSelfPermission(WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(WRITE_EXTERNAL_STORAGE);
            }
            if (checkSelfPermission(READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(READ_EXTERNAL_STORAGE);
            }
            if (!permissions.isEmpty()) {
                requestPermissions(permissions.toArray(new String[0]), REQUEST_ID_MULTIPLE_PERMISSIONS);
            }
        }
        if (permissions.isEmpty()) {
            cameraBridgeViewBase.enableView();
            initObjectDetection();
        }
    }

    private void showDialogOK(String message, DialogInterface.OnClickListener okListener) {
        new AlertDialog.Builder(this)
                .setMessage(message)
                .setPositiveButton("OK", okListener)
                .setNegativeButton("Cancel", okListener)
                .create()
                .show();
    }

    public void initObjectDetection() {

        if (!netInitialized) {

            String path = getApplicationInfo().dataDir;
            File directory = new File(path);
            File[] files = directory.listFiles((dir, name) -> name.contains("."));

            if (files.length != 3) {
                downloadModelsFromFirebase();

            } else {
                for (File file : files) {
                    if ("coco.names".equals(file.getName())) {
                        try {
                            loadNamesOfClasses(new FileInputStream(file));
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                    } else if ("yolov3.cfg".equals(file.getName())) {
                        this.yoloCfg = file.getPath();
                    } else {
                        this.yoloWeights = file.getPath();
                    }
                }
                if (yoloCfg != null && yoloWeights != null && cocoNames != null && !cocoNames.isEmpty()) {
                    initializeNet();

                } else {
                    finish();
                }
            }
            Log.d(TAG, "STARTED FROM CREATE - FIRST TIME");
        }

    }

    private void downloadModelsFromFirebase() {
        downloadNames();
    }

    private void downloadNames() {
        StorageReference namesRef = this.mStorageRef.child("coco.names");

        File localFile;
        //localFile = File.createTempFile("coco_", ".names");
        localFile = new File(getApplicationInfo().dataDir + File.separator + "coco.names");

        File finalLocalFile = localFile;
        namesRef.getFile(localFile)
                .addOnSuccessListener(taskSnapshot -> {
                    Log.d("NAMES_DOWNLOAD", "File downloaded");
                    FileInputStream namesStream = null;
                    try {
                        namesStream = new FileInputStream(finalLocalFile);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                    assert namesStream != null;
                    loadNamesOfClasses(namesStream);
                    downloadConfig();
                })
                .addOnFailureListener(exception -> Log.d("NAMES_DOWNLOAD", "File not downloaded"));
    }

    private void downloadConfig() {
        StorageReference namesRef = this.mStorageRef.child("yolov3.cfg");

        File localFile;
        //localFile = File.createTempFile("yolov3_", ".cfg");
        localFile = new File(getApplicationInfo().dataDir + File.separator + "yolov3.cfg");

        File finalLocalFile = localFile;
        namesRef.getFile(localFile)
                .addOnSuccessListener(taskSnapshot -> {
                    Log.d("CONFIG_DOWNLOAD", "File downloaded");
                    this.yoloCfg = finalLocalFile.getPath();
                    downloadWeights();
                })
                .addOnFailureListener(exception -> Log.d("CONFIG_DOWNLOAD", "File not downloaded"));
    }

    private void downloadWeights() {
        StorageReference namesRef = this.mStorageRef.child("yolov3.weights");

        File localFile;
        //localFile = File.createTempFile("yolov3_", ".weights");
        localFile = new File(getApplicationInfo().dataDir + File.separator + "yolov3.weights");

        File finalLocalFile = localFile;
        namesRef.getFile(localFile)
                .addOnSuccessListener(taskSnapshot -> {
                    Log.d("WEIGHTS_DOWNLOAD", "File downloaded");
                    this.yoloWeights = finalLocalFile.getPath();
                    initializeNet();
                })
                .addOnFailureListener(exception -> Log.d("WEIGHTS_DOWNLOAD", "File not downloaded"));
    }

    private void initializeNet() {
        // Get and initialize the corresponding network from Darknet
        net = Dnn.readNetFromDarknet(yoloCfg, yoloWeights);
        netInitialized = true;
    }

    private void loadNamesOfClasses(FileInputStream namesStream) {
        BufferedReader reader;
        this.cocoNames = new ArrayList<>();
        try {
            reader = new BufferedReader(new FileReader(namesStream.getFD()));
            String line = reader.readLine();
            while (line != null) {
                this.cocoNames.add(line);
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
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
