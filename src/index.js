import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import Util from './Util';

// Off warning
// export TF_CPP_MIN_LOG_LEVEL=3

const FILE_FOLDER_PATH = './src/models/train/';

const MODEL_PATH = './src/models/AON_MODEL.model';

const ITERATION = 3;

async function App() {

    // Create the classifier.
    const classifier = knnClassifier.create();

    // Load mobilenet.
    const mobilenet = await mobilenetModule.load();

    // Get train datas
    const trainDatas = fs.readdirSync(FILE_FOLDER_PATH)
        .filter(file => file !== '.DS_Store')
        .map(file => ({
            label: file.split('.')[0],
            image: tf.node.decodeImage(fs.readFileSync(path.join(FILE_FOLDER_PATH, file)), 3),
        }));

    // Add classifier
    Array.from({ length: ITERATION }).forEach(() => {
        trainDatas.forEach(target => {
            const { label, image } = target;
            const logits = mobilenet.infer(image, true);
            classifier.addExample(logits, label);
        });
    })

    // Save dataset
    const dataset = classifier.getClassifierDataset();
    fs.writeFileSync(MODEL_PATH, Util.modelSave(dataset));
}

App();
