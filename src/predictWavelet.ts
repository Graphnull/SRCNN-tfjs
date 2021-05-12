import * as tf from '@tensorflow/tfjs-node'
import sharp from 'sharp'
import fs from 'fs'

let inputPath = './test.jpg';
let outputPath = '4result.png'

let constructImage = async (lImg:Float32Array,wavelets:Float32Array,width:number,height:number, path:string)=>{

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
let constructImageBase = async (wavelets:Float32Array,width:number,height:number, path:string)=>{

    let img = new Float32Array(width*2*height*2*3);

    for(let c =0;c!==3;c++){
        for(let y=0;y!==height;y++){
            for(let x=0;x!==width;x++){
                let avg1 = (128 +wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                let avg2 = (128 -wavelets[width*height*3*c+ y*width*3+x*3+2]/2)/2
                img[((y*2+0)*width*2+x*2+0)*3+c]=avg1+wavelets[width*height*3*c+ y*width*3+x*3+0]/2
                img[((y*2+0)*width*2+x*2+1)*3+c]=avg1-wavelets[width*height*3*c+ y*width*3+x*3+0]/2
                img[((y*2+1)*width*2+x*2+0)*3+c]=avg2+wavelets[width*height*3*c+ y*width*3+x*3+1]/2
                img[((y*2+1)*width*2+x*2+1)*3+c]=avg2-wavelets[width*height*3*c+ y*width*3+x*3+1]/2
            }
        }
    }

    
    let buf = Buffer.from(img.map(v=> Math.min(Math.max(v,0),255)))

    await sharp(buf,{raw:{width:width*2, height:height*2, channels: 3}}).png().toFile(path+'base.png')
}
let main = async ()=>{
let model
    try{
        model = await tf.loadLayersModel('file://./modelsWavelet/'+fs.readdirSync('./modelsWavelet').sort((l,r)=>parseFloat(l.split('_')[1])-parseFloat(r.split('_')[1]))[0]+'/model.json');
    }catch(err){
console.log('111', err, './modelsWavelet/'+fs.readdirSync('./modelsWavelet').sort((l,r)=>parseFloat(l.split('_')[1])-parseFloat(r.split('_')[1]))[0]+'/model.json');
process.exit();
    }
    (model.inputs[0] as any).shape=[null,null,null,1]
    model.inputs[0].sourceLayer.batchInputShape = [null,null,null,1]

    let info = await sharp(inputPath).metadata();
    let {width, height} = info;
    if(!height||!width){
        throw new Error('asd')
    }

    let orig = sharp(inputPath).clone().toColourspace('lab');
    let img1 = Buffer.concat([
        await orig.clone().extractChannel(0).toColourspace('b-w').raw().toBuffer(),
        await orig.clone().extractChannel(1).toColourspace('b-w').raw().toBuffer(),
        await orig.clone().extractChannel(2).toColourspace('b-w').raw().toBuffer(),
    ])

    let inp = tf.tensor4d(img1,[3,height,width,1]).toFloat().div(255);
    let res = model.predict(inp) as tf.Tensor3D;


    await constructImage(
        inp.mul(255).dataSync()as Float32Array,
        res.mul(255).dataSync()as Float32Array,
        width,height, outputPath
    )
    await constructImageBase(
    res.mul(255).dataSync()as Float32Array,
    width,height, outputPath
)
    // await sharp(new Buffer(res.mul(255).maximum(0).minimum(255).dataSync()),{raw:{width:info.width*2, height:info.height*2,channels:3}})
    // .png()
    // .toFile(outputPath)
   

}

main();