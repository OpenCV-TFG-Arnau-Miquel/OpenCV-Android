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
    private Brightness oldLightState;
    public boolean notableChange;

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

    public void setLightState(Brightness lightState) {
        synchronized (this) {
            oldLightState = this.lightState;
            this.lightState = lightState;
            changeLight();
        }
    }

    public void changeLight() {
        if (isDark(this.oldLightState) && isLight(this.lightState)) {
            notableChange(true);
        } else if (isLight(this.oldLightState) && isDark(this.lightState)){
            notableChange(true);
        } else {
            notableChange(false);
        }
    }

    public void notableChange(boolean change) {
        this.notableChange = change;
    }

    public boolean isNotableChange() {
        return notableChange;
    }

    public boolean isLight(Brightness light) {
        return light.equals(Brightness.LIGHT);
    }

    public boolean isDark(Brightness light) {
        return light.equals(Brightness.DARK);
    }
}
