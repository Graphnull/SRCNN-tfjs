import * as tf from '@tensorflow/tfjs-node';
import {modelL, modelC} from './model'
import fs from 'fs'
import sharp from 'sharp'

let imagesPath = '/media/eva/124571ec-9959-4bad-b97f-656bce223877/datasets/ukbench/full/';

let files = fs.readdirSync(imagesPath)
console.log('files: ', files.length);

let validataionFiles = files.slice(3500, 3900);
files = files.slice(0, 3500);

let constructImage = async (lImg:Buffer,cImg1:Buffer,cImg2:Buffer,width:number,height:number, path:string)=>{

    let img = Buffer.alloc(width*height*3);
    
    let r = cImg1.map((v,i)=>lImg[i]+(v-127)*2);
    let g = lImg;
    let b = cImg2.map((v,i)=>lImg[i]+(v-127)*2, width*height*2);
    for(let i=0;i!==width*height;i++){
        img[i*3+0] = r[i]
        img[i*3+1] = g[i]
        img[i*3+2] = b[i]
    }
    // await sharp(Buffer.from(cImg1.map((v,i)=>(v-127)*2+lImg[i])),{raw:{width, height, channels: 1}})
    // .joinChannel(lImg)
    // .joinChannel(Buffer.from(cImg2.map((v,i)=>(v-127)*2+lImg[i], width*height*2)))
    // .png().toFile(path)
    await sharp(img,{raw:{width, height, channels: 3}}).png().toFile(path)
}
let getImages = async (folder: string, files: string[], index: number, batchSize: number) => {

    let info = await sharp(folder + files[index]).metadata();
    let width = info.width || 0;
    let height = info.height || 0;
    let origImageL :Buffer[]= [];
    let origImageC :Buffer[]= [];
    let resizedImageL :Buffer[]= [];
    let resizedImageC :Buffer[]= [];
    for (let i = 0; i !== batchSize; i++) {
        let file = sharp(folder + files[index + i]);
        let orig = file.clone().toColourspace('lab').resize(width / 2, height / 2)
        let img1 = await orig.clone().extractChannel(1).toColourspace('b-w').raw().toBuffer()
        origImageL.push(img1);
        

        origImageC.push(Buffer.from((await orig.clone().extractChannel(0).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-img1[i])/2+127 )));
        origImageC.push(Buffer.from((await orig.clone().extractChannel(2).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-img1[i])/2+127 )));

        //constructImage(origImageL[origImageL.length-1], origImageC[origImageC.length-2], origImageC[origImageC.length-1], width/2, height/2, './test3.png')

        let resized = file.clone().toColourspace('lab').resize(width / 4, height / 4)
        let imgr = await resized.extractChannel(1).toColourspace('b-w').raw().toBuffer();
        resizedImageL.push(imgr);
        resizedImageC.push(Buffer.from((await resized.extractChannel(0).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-imgr[i])/2+127 )));
        resizedImageC.push(Buffer.from((await resized.extractChannel(2).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-imgr[i])/2+127 )));
        
    }
    return { origImageL: Buffer.concat(origImageL), origImageC: Buffer.concat(origImageC), 
        resizedImageL: Buffer.concat(resizedImageL), resizedImageC: Buffer.concat(resizedImageC), 
        width:width/2, height: height/2 }
}

let getValidation = async () => {


    let errorL = tf.scalar(0);
    let errorC = tf.scalar(0);
    for (let i = 0; i !== validataionFiles.length; i++) {

        let { origImageC, origImageL, resizedImageC, resizedImageL, width, height } = await getImages(imagesPath, validataionFiles, i, 1)

        let count = origImageL.length / (height * width * 1);
        errorL = errorL.add(tf.tidy(() => {
            let y = tf.tensor4d(origImageL, [count, height, width, 1])

            let x = tf.tensor4d(resizedImageL, [count, height / 2, width / 2, 1])

            let result = modelL.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        }))
        errorC = errorC.add(tf.tidy(() => {
            let y = tf.tensor4d(origImageC, [count*2, height, width, 1])

            let x = tf.tensor4d(resizedImageC, [count*2, height / 2, width / 2, 1])

            let result = modelC.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        }))
       
    }
    return [errorL.div(validataionFiles.length).dataSync()[0], errorC.div(validataionFiles.length*2).dataSync()[0]];
}

let minimize = (origImageL: Buffer, origImageC: Buffer, resizedImageL: Buffer, resizedImageC: Buffer, width: number, height: number) => {
    let count = origImageL.length / (height * width * 1)
    tf.tidy(() => {

        let time = new Date().valueOf();
        optimizerL.minimize(() => {
            let y = tf.tensor4d(origImageL, [count, height, width, 1])
            let x = tf.tensor4d(resizedImageL, [count, height / 2, width / 2, 1])
            let result = modelL.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        })


        optimizerC.minimize(() => {
            let y = tf.tensor4d(origImageC, [count*2, height, width, 1])
            let x = tf.tensor4d(resizedImageC, [count*2, height / 2, width / 2, 1])
            let result = modelC.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        })
        console.log('sd', new Date().valueOf()-time);
    })
}
let optimizerL = tf.train.adam(3.0e-4);
let optimizerC = tf.train.adam(3.0e-4);
let loss = tf.losses.meanSquaredError;

let main = async () => {
    let epochs = 300;
    let batchSize = 16;
    for (let e = 0; e !== epochs; e++) {

        for (let i = 0; i !== Math.floor(files.length / batchSize) * batchSize; i += batchSize) {

            console.log(i);
            let { origImageL,origImageC, resizedImageL,resizedImageC, width, height } = await getImages(imagesPath, files, i, batchSize);
            minimize(origImageL, origImageC, resizedImageL, resizedImageC, width, height);

        }
        let shuffle = (array: string[]) => array.sort(() => Math.random() - 0.5);
        files = shuffle(files)
        let valError = await getValidation()
        console.log('valError: ', valError);
        modelL.save(`file://./modelsL/${e}_${valError[0]}`)
        modelC.save(`file://./modelsC/${e}_${valError[1]}`)
    }

}
main();
