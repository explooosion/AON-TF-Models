import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import Util from '../src/Util';
import DataBase from '../src/DataBase';

// Off warning
// export TF_CPP_MIN_LOG_LEVEL=3

const FILE_FOLDER_PATH = './src/models/train/';

const MODEL_PATH = './src/models/AON_MODEL.model';

async function App() {

    // Create the classifier.
    const dataset = fs.readFileSync(MODEL_PATH);
    const classifier = knnClassifier.create();
    classifier.setClassifierDataset(Util.modelLoad(dataset));

    // Load mobilenet.
    const mobilenet = await mobilenetModule.load();

    // Make a prediction.
    const TARGET = 'the_art_of_war_forest_chapter.png';
    const pre_img = tf.node.decodeImage(fs.readFileSync(path.join(FILE_FOLDER_PATH, TARGET)), 3);
    const xlogits = mobilenet.infer(pre_img, 'conv_preds');

    const result = await classifier.predictClass(xlogits);
    console.log('Predictions:', DataBase.find(d => d.id === result.label));
}

App();
