package com.example.androidtfg;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;

class AccelerometerListener implements SensorEventListener {

    private boolean highMovement = false;

    @Override
    public void onSensorChanged(SensorEvent event) {

        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            EventValues eventValues = new EventValues(event).calculateValues();
            int xmin = eventValues.getXmin();
            int xmax = eventValues.getXmax();
            int ymin = eventValues.getYmin();
            int ymax = eventValues.getYmax();
            int zmin = eventValues.getZmin();
            int zmax = eventValues.getZmax();

            if ((xmin > 0 && xmax < 0) &&
                (ymin > 0 && ymax < 0) &&
                (zmin > 0 && zmax < 0)) {

                synchronized (this) {
                    this.highMovement = false;
                }
            } else {

                synchronized (this) {
                    this.highMovement = true;
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    public boolean isHighMovement() {
        synchronized (this) {
            return highMovement;
        }
    }

    private class EventValues {
        float x1 = -0.08f;
        float x2 = 0.3f;
        float y1 = -0.5f;
        float y2 = 0.5f;
        float z1 = -0.8f;
        float z2 = 0.08f;

        private SensorEvent event;
        private int xmin;
        private int xmax;
        private int ymin;
        private int ymax;
        private int zmin;
        private int zmax;

        public EventValues(SensorEvent event) {
            this.event = event;
        }

        public int getXmin() {
            return xmin;
        }

        public int getXmax() {
            return xmax;
        }

        public int getYmin() {
            return ymin;
        }

        public int getYmax() {
            return ymax;
        }

        public int getZmin() {
            return zmin;
        }

        public int getZmax() {
            return zmax;
        }

        public EventValues calculateValues() {
            float x = event.values[0];
            xmin = Float.compare(x, x1);
            xmax = Float.compare(x, x2);
            float y = event.values[1];
            ymin = Float.compare(y, y1);
            ymax = Float.compare(y, y2);
            float z = event.values[2];
            zmin = Float.compare(z, z1);
            zmax = Float.compare(z, z2);
            return this;
        }
    }
}
