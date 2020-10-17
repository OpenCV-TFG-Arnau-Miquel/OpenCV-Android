package com.example.androidtfg;

import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.AsyncTask;
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
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

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

    public static final int REQUEST_ID_MULTIPLE_PERMISSIONS = 3;
    private static final String TAG = "MainActivity";
    private static final int EQ = 10000;
    private CameraBridgeViewBase cameraBridgeViewBase;
    private Net net;
    private String yoloCfg;
    private String yoloWeights;
    private StorageReference mStorageRef;
    private ArrayList<String> cocoNames;
    private boolean netInitialized = false;
    private ObjectDetectionTask objectDetectionTask;
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private AccelerometerListener accelerometerListener;
    private boolean running = false;
    private Mat oldFrame;

    private ArrayList<Detection> detectionsDone = new ArrayList<>();

    private int frameCounter = 0;
    private long lastTaskTime = 0;
    private boolean forcedNotEq = false;
    private boolean forcedEq = false;
    private boolean previousStateEquals = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.accelerometerListener = new AccelerometerListener();
        this.sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        this.accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        mStorageRef = FirebaseStorage.getInstance().getReference();

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There's a problem", Toast.LENGTH_SHORT).show();
        }

        oldFrame = new Mat();

    }

    @Override
    protected void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
    }

    @Override
    protected void onResume() {
        Log.d(TAG, "onResume");
        super.onResume();
        requestAppPermissions();
        sensorManager.registerListener(accelerometerListener, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);


    }

    @Override
    protected void onPause() {
        Log.d(TAG, "onPause");

        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
        if (objectDetectionTask != null && !objectDetectionTask.isCancelled() &&
                !objectDetectionTask.getStatus().equals(AsyncTask.Status.FINISHED)) {
            objectDetectionTask.cancel(true);
        }
        sensorManager.unregisterListener(accelerometerListener);

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

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat newFrame = inputFrame.rgba();
        Mat greyNewFrame = inputFrame.gray();

        formatFrame(newFrame);
        formatFrame(greyNewFrame);

        if (frameCounter > 30) {

            boolean areSimilarFrames = false;
            if (newFrame != null && !oldFrame.empty()) {
                areSimilarFrames = compareFrames(oldFrame, greyNewFrame);
            }

            Log.d(TAG, "ARE SIMILAR =                    " + areSimilarFrames);

            greyNewFrame.copyTo(oldFrame);

            if (objectDetectionTask != null &&
                    (objectDetectionTask.isCancelled() || objectDetectionTask.getStatus().equals(AsyncTask.Status.FINISHED))) {
                forcedNotEq = false;
                forcedEq = false;
            }


            if (netInitialized && !accelerometerListener.isHighMovement()) {

                if (!areSimilarFrames) {
                    if (previousStateEquals) {
                        newTask(newFrame);  // we stop other detections from equality loop to start a new one.
                        detectionsDone.clear(); // this frame is different from the previous ones so we delete their detections
                        forcedEq = false;
                        forcedNotEq = true; // this detection must finish.
                        lastTaskTime = System.currentTimeMillis();
                    }
                    // start detections for supposed changing loop (image constantly changing)
                    if (!forcedNotEq) { // wait til the forced detection is done
                        newTask(newFrame);  // we stop other detections to start a new one.
                        forcedNotEq = true; // this detection must finish.
                        lastTaskTime = System.currentTimeMillis();
                        // this will work as sequential detection for an image constantly changing.
                        // Putting time on this can make the detection to be done in a more wide lapse of time
                        // and loose more different frames
                    }

                } else {
                    if (!previousStateEquals) {
                        newTask(newFrame);  // we stop other detections from difference loop to start a new one.
                        detectionsDone.clear(); // this frame is equal from the previous one, but detection can come from much later frames
                        forcedNotEq = false;
                        forcedEq = true; // this first detection for a supposed loop of equal frames must finish.
                        lastTaskTime = System.currentTimeMillis();
                    } else { // we are in a loop of equal frames
                        if (mustDetect()) { // every 10 sec recalculate
                            newTask(newFrame);  // we stop other detections to start a new one.
                            lastTaskTime = System.currentTimeMillis();
                        } else {    // it does not need recalculation yet
                            if (detectionsDone.isEmpty()) { // can be an error
                                if (!forcedEq) {    // if no previous forced detection is working
                                    newTask(newFrame); // we stop other detections to start a new one.
                                    forcedEq = true;    // this is a forced detection - must finish.
                                    lastTaskTime = System.currentTimeMillis();
                                }
                            }
                        }
                    }
                }


            } else if (accelerometerListener.isHighMovement()) {
                synchronized (this) {
                    detectionsDone.clear();
                    if (objectDetectionTask != null && !objectDetectionTask.isCancelled() &&
                            !objectDetectionTask.getStatus().equals(AsyncTask.Status.FINISHED)) {
                        objectDetectionTask.cancel(true);
                    }
                }
            }

            synchronized (this) {
                drawDetections(newFrame);
            }

            previousStateEquals = areSimilarFrames;

        } else {
            frameCounter++;
        }
        return newFrame;
    }

    private void newTask(Mat newFrame) {
        boolean canceled = false;
        if (objectDetectionTask != null && !objectDetectionTask.isCancelled() && !objectDetectionTask.getStatus().equals(AsyncTask.Status.FINISHED)) {
            Log.d(TAG, "CANCEL TASK");
            canceled = true;
            objectDetectionTask.cancel(true);
        }
        if (!canceled) Log.d(TAG, "NOT CANCEL TASK");
        initTask(newFrame);
    }

    private void initTask(Mat newFrame) {
        objectDetectionTask = new ObjectDetectionTask(net, this);
        Log.d(TAG, "NEW TASK");
        objectDetectionTask.execute(newFrame);
    }

    private boolean mustDetect() {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastTaskTime >= EQ) {
            Log.d(TAG, "Must detect");
            return true;
        }
        return false;
    }


    private boolean compareFrames(Mat oldFrame, Mat newFrame) {
        Mat difference = new Mat();
        Core.compare(oldFrame, newFrame, difference, Core.CMP_EQ);
        int count = Core.countNonZero(difference);
        return count >= 15000;
    }

    private void formatFrame(Mat frame) {
        Core.flip(frame, frame, 0);
        Core.flip(frame, frame, 1);
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
        List<String> permissions = new ArrayList<>();
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
        }

    }

    private void downloadModelsFromFirebase() {
        downloadNames();
    }

    private void downloadNames() {
        StorageReference namesRef = this.mStorageRef.child("coco.names");

        File localFile;
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

    private void drawDetections(Mat newFrame) {
        if (detectionsDone != null && !detectionsDone.isEmpty()) {
            for (Detection detection : detectionsDone) {
                Imgproc.putText(newFrame, cocoNames.get(detection.getId()).toUpperCase() + " " + detection.getIntConf() + "%", detection.getTextPosition(), Core.FONT_HERSHEY_COMPLEX, 0.75, new Scalar(255, 255, 0), 1);
                Imgproc.rectangle(newFrame, detection.getBox().tl(), detection.getBox().br(), new Scalar(255, 0, 0), 2);
            }
        }
    }

    public void setNewDetections(ArrayList<Detection> detections) {
        synchronized (this) {
            detectionsDone.clear();
            this.detectionsDone.addAll(detections);
        }
    }
}
