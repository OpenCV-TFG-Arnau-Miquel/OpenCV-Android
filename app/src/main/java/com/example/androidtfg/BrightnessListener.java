package com.example.androidtfg;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;

public class BrightnessListener implements SensorEventListener {

    enum Brightness {
        DARK,
        MID,
        LIGHT
    }

    private Brightness lightState;

    @Override
    public void onSensorChanged(SensorEvent event) {

        //Aqui caldria posar un estat de inici, si canvia dels tipus de brightness
        //suposarem que pot ser un canvi significant
        //No esta testejat falta establir varems 
        if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
            if (event.values[0] <= 0.0f) {
                setLightState(Brightness.DARK);
            } else if (event.values[0] <= 1000.0f) {
                setLightState(Brightness.MID);
            } else if (event.values[0] > 1000.0f) {
                setLightState(Brightness.LIGHT);
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    public Brightness getLightState() {
        return lightState;
    }

    public void setLightState(Brightness lightState) {
        synchronized (this) {
            lightState = lightState;
        }
    }

    public boolean isDark() {
        return getLightState().equals(Brightness.DARK);
    }
}
