# ConditionalGAN_birds_model
Code to train model in ConditionalGAN architecture based on CUB_200_2011 dataset

## Running
Python version: 3.8, all needed pip packages should be in the requirements file so you can use

```shell
  pip install -r requirements.txt
```

If you want to run this on the same files as I did, you can just run dependencies.py file. It will download reformatted dataset, captions and sentence encoder.

But if you want to reformat data by yourself, you need to follow these steps:

To start training process you need to save [dataset](https://data.caltech.edu/records/65de6-vp158), [captions](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view?resourcekey=0-sZrhftoEfdvHq6MweAeCjA), [sentence encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/tensorFlow2/variations/universal-sentence-encoder/versions/2?tfhub-redirect=true) and [yolov3](https://pjreddie.com/darknet/yolo/) into proper catalogues 

Whole tree should look like this:

![image](https://github.com/rombii/ConditionalGAN_birds_model/assets/46005468/be74df53-b4e3-46fb-99fd-97deec91c6f9)

after that you need to reformat dataset, check if any unwanted images didn't pass through and run main.py file.
