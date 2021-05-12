import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs'
import sharp from 'sharp'

let model = tf.sequential();
model.add(tf.layers.conv2d({ filters: 128, kernelSize: [9, 9], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true, inputShape: [120,160, 1]}));
model.add(tf.layers.leakyReLU());
model.add(tf.layers.conv2d({ filters: 32, kernelSize: [5, 5], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
model.add(tf.layers.leakyReLU());
model.add(tf.layers.conv2d({ filters: 16, kernelSize: [3, 3], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
model.add(tf.layers.leakyReLU());
model.add(tf.layers.conv2d({ filters: 3, kernelSize: [3, 3], kernelInitializer: 'glorotUniform', activation: 'tanh', padding: 'same', useBias: true }));


model.summary();

let imagesPath = '/media/eva/124571ec-9959-4bad-b97f-656bce223877/datasets/ukbench/full/';

let files = fs.readdirSync(imagesPath)
console.log('files: ', files.length);

let validataionFiles = files.slice(3500, 3900);
files = files.slice(0, 3500);

let constructImage = async (lImg:Float32Array|Buffer,wavelets:Float32Array|Buffer,width:number,height:number, path:string)=>{

    let img = new Float32Array(width*2*height*2*3);

    for(let c =0;c!==3;c++){
        for(let y=0;y!==height;y++){
            for(let x=0;x!==width;x++){
                // let avg1 = (128 +wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                // let avg2 = (128 -wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                let avg1 = (lImg[width*height*c+ y*width+x]*2 +wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                let avg2 = (lImg[width*height*c+ y*width+x]*2 -wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                img[((y*2+0)*width*2+x*2+0)*3+c]=avg1+wavelets[width*height*3*c+ y*width*3+x*3+0]/2
                img[((y*2+0)*width*2+x*2+1)*3+c]=avg1-wavelets[width*height*3*c+ y*width*3+x*3+0]/2
                img[((y*2+1)*width*2+x*2+0)*3+c]=avg2+wavelets[width*height*3*c+ y*width*3+x*3+1]/2
                img[((y*2+1)*width*2+x*2+1)*3+c]=avg2-wavelets[width*height*3*c+ y*width*3+x*3+1]/2
            }
        }
    }

    
    let buf = Buffer.from(img.map(v=> Math.min(Math.max(v,0),255)))

    await sharp(buf,{raw:{width:width*2, height:height*2, channels: 3}}).png().toFile(path)
}
function Float32Concat(array: Float32Array[])
{
    var firstLength = 0;
    array.forEach(v=>{firstLength+=v.length})
    let result = new Float32Array(firstLength);
    let offset = 0;
    array.forEach(v=>{result.set(v, offset);offset+=v.length})

    return result;
}
let getImages = async (folder: string, files: string[], index: number, batchSize: number) => {

    let info = await sharp(folder + files[index]).metadata();
    let width = info.width || 0;
    let height = info.height || 0;
    let resizedImages :Buffer[]= [];
    let waveletLevels :Float32Array[]= [];
    for (let i = 0; i !== batchSize; i++) {
        let file = sharp(folder + files[index + i]);
        let orig = file.clone().toColourspace('lab').resize(width / 2, height / 2)

        for(let c =0;c!==3;c++){

            let image = await orig.clone().extractChannel(c).toColourspace('b-w').raw().toBuffer()
            let waveletLevel = new Float32Array((width / 4)*(height / 4)*3)
            let resizedImage = Buffer.alloc((width / 4)*(height / 4))

            for(let y=0;y!==height / 4;y++){
                for(let x=0;x!==width / 4;x++){
                    waveletLevel[(y*(width/4)+x)*3+0] = image[(y*2+0)*(width/2)+(x*2+0)]-image[(y*2+0)*(width/2)+(x*2+1)]
                    let avgx1 = (image[(y*2+0)*(width/2)+(x*2+0)]+image[(y*2+0)*(width/2)+(x*2+1)])
                    waveletLevel[(y*(width/4)+x)*3+1] = image[(y*2+1)*(width/2)+(x*2+0)]-image[(y*2+1)*(width/2)+(x*2+1)]
                    let avgx2 = (image[(y*2+1)*(width/2)+(x*2+0)]+image[(y*2+1)*(width/2)+(x*2+1)])
                    waveletLevel[(y*(width/4)+x)*3+2] = avgx1-avgx2
                    resizedImage[(y*(width/4)+x)] = (avgx1/2+avgx2/2)/2

                }
            }
            resizedImages.push(resizedImage);
            waveletLevels.push(waveletLevel);
        }

        
    }
    return { resizedImages: Buffer.concat(resizedImages), waveletLevels:Float32Concat(waveletLevels), 
        width:width/4, height: height/4 }
}

let getValidation = async () => {


    let errorL = tf.scalar(0);
    let errorC = tf.scalar(0);
    for (let i = 0; i !== validataionFiles.length; i++) {

        let { resizedImages, waveletLevels, width, height } = await getImages(imagesPath, validataionFiles, i, 1)

        let count = resizedImages.length / (height * width * 1);
        errorL = errorL.add(tf.tidy(() => {
            let y = tf.tensor4d(waveletLevels, [count, height, width, 3])

            let x = tf.tensor4d(resizedImages, [count, height, width, 1])

            let result = model.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        }))
    }
    return [errorL.div(validataionFiles.length).dataSync()[0], errorC.div(validataionFiles.length*2).dataSync()[0]];
}

let minimize = (waveletLevels: Float32Array, resizedImages: Buffer, width: number, height: number) => {
    let count = resizedImages.length / (height * width * 1)
    tf.tidy(() => {
        let time = new Date().valueOf();
        optimizer.minimize(() => {
            let y = tf.tensor4d(waveletLevels, [count, height, width, 3])
            let x = tf.tensor4d(resizedImages, [count, height, width, 1])
            let result = model.predict(x.div(255)) as tf.Tensor3D;
            return loss(result, y.div(255))
        })
        console.log('sd', new Date().valueOf()-time);
    })
}
let optimizer = tf.train.adam(3.0e-4);
let loss = tf.losses.meanSquaredError;

let main = async () => {
    let epochs = 300;
    let batchSize = 16;
    for (let e = 0; e !== epochs; e++) {

        for (let i = 0; i !== Math.floor(files.length / batchSize) * batchSize; i += batchSize) {

            console.log(i);
            let { waveletLevels, resizedImages, width, height } = await getImages(imagesPath, files, i, batchSize);
            minimize(waveletLevels, resizedImages, width, height);

        }
        let shuffle = (array: string[]) => array.sort(() => Math.random() - 0.5);
        files = shuffle(files)
        let valError = await getValidation()
        console.log('valError: ', valError);
        model.save(`file://./modelsWavelet/${e}_${valError[0]}`)
    }

}
main();
