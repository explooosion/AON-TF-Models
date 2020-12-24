import * as tf from '@tensorflow/tfjs-node';

export default class Util {

    /**
     * Convert dataset to string
     * @param {*} dataset classifier dataset
     */
    static modelSave(dataset) {
        return JSON.stringify(Object.entries(dataset).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    }

    /**
     * Convert string to dataset
     * @param {*} str model string
     */
    static modelLoad(str) {
        return Object.fromEntries(JSON.parse(str).map(([label, data, shape]) => [label, tf.tensor(data, shape)]));
    }
}