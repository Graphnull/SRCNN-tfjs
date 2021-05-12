import * as tf from '@tensorflow/tfjs-node'
import sharp from 'sharp'
import fs from 'fs'

let inputPath = './test.jpg';
let outputPath = '4result.png'

let constructImage = async (lImg:Float32Array,cImg1:Float32Array,cImg2:Float32Array,width:number,height:number, path:string)=>{

    let img = Buffer.alloc(width*height*3);
    
    let r = cImg1.map((v,i)=>lImg[i]+(v-127)*2);
    let g = lImg;
    let b = cImg2.map((v,i)=>lImg[i]+(v-127)*2, width*height*2);
    for(let i=0;i!==width*height;i++){
        img[i*3+0] = Math.min(Math.max(r[i],0),255)
        img[i*3+1] = Math.min(Math.max(g[i],0),255)
        img[i*3+2] = Math.min(Math.max(b[i],0),255)
    }
    // await sharp(Buffer.from(cImg1.map((v,i)=>(v-127)*2+lImg[i])),{raw:{width, height, channels: 1}})
    // .joinChannel(lImg)
    // .joinChannel(Buffer.from(cImg2.map((v,i)=>(v-127)*2+lImg[i], width*height*2)))
    // .png().toFile(path)
    await sharp(img,{raw:{width, height, channels: 3}}).png().toFile(path)
}

let main = async ()=>{
let modelL
    try{
        modelL = await tf.loadLayersModel('file://./modelsL/'+fs.readdirSync('./modelsL').sort((l,r)=>parseFloat(l.split('_')[1])-parseFloat(r.split('_')[1]))[0]+'/model.json');
    }catch(err){
console.log('111', err, './modelsL/'+fs.readdirSync('./modelsL').sort((l,r)=>parseFloat(l.split('_')[1])-parseFloat(r.split('_')[1]))[0]+'/model.json');
process.exit();
    }
    (modelL.inputs[0] as any).shape=[null,null,null,1]
    modelL.inputs[0].sourceLayer.batchInputShape = [null,null,null,1]
    let modelC
    try{
        modelC = await tf.loadLayersModel('file://./modelsC/'+fs.readdirSync('./modelsC').sort((l,r)=>parseFloat(l.split('_')[1])-parseFloat(r.split('_')[1]))[0]+'/model.json');
    }catch(err){
        console.log('err2: ', err);
        process.exit();
    }
    (modelC.inputs[0] as any).shape=[null,null,null,1]
    modelC.inputs[0].sourceLayer.batchInputShape = [null,null,null,1]

    let info = await sharp(inputPath).metadata();
    let {width, height} = info;
    if(!height||!width){
        throw new Error('asd')
    }

    let orig = sharp(inputPath).clone().toColourspace('lab');
    let img1 = await orig.clone().extractChannel(1).toColourspace('b-w').raw().toBuffer()

    //let data = await sharp(inputPath).removeAlpha().raw().toBuffer()
    let inp = tf.tensor4d(img1,[1,height,width,1]).toFloat().div(255);
    let res = modelL.predict(inp) as tf.Tensor3D;


    let dif1 = tf.tensor4d((await orig.clone().extractChannel(0).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-img1[i])/2+127 ),[1,height,width,1]).toFloat().div(255);
    let dif2 = tf.tensor4d((await orig.clone().extractChannel(2).toColourspace('b-w').raw().toBuffer()).map((v,i)=>(v-img1[i])/2+127 ),[1,height,width,1]).toFloat().div(255);


    let time = new Date().valueOf();
    let res1 = modelC.predict(dif1) as tf.Tensor3D;
    let res2 = modelC.predict(dif2) as tf.Tensor3D;

    console.log('sd', new Date().valueOf()-time);
    await constructImage(
        res.mul(255).dataSync()as Float32Array,
        res1.mul(255).dataSync()as Float32Array,
        res2.mul(255).dataSync()as Float32Array,
        width*2,height*2, outputPath
    )
    // await sharp(new Buffer(res.mul(255).maximum(0).minimum(255).dataSync()),{raw:{width:info.width*2, height:info.height*2,channels:3}})
    // .png()
    // .toFile(outputPath)
   

}

main();