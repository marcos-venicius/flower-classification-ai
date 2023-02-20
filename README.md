# Flower classification

This is an AI to classificate flower category like: daisy, sunflower, tulip...

## Disclaimer

I'm not an AI developer, just trying to code some AI to test my brain.

You can improve this! just submit a pull request and let's go learn.

## How to run this code?

when you run at first time i recommend you to use `VIEW_RESULTS=1` flag.
this will show a graph at the end of process with the training results


after first running, this will have a model.fit cache in `data` folder, so you can run 
`./main.py ./images/sunflower.jpg` the AI will try to predict what is this flower category.

## Obs

I have no idea why using the `rose.jpg` image it's returing that is a `tulip` lol

## Troubleshooting

- cannot run `./main.py`
  ```shell
  chmod u+x ./main.py
  ```
- cannot see graphs
  ```shell
  sudo apt install python3-tk
  ```
- not have tensorflow
  ```shell
  pip3 install tensorflow
  ```
- not have PIL
  ```shell
  pip3 install pillow
  ```
- not have matplotlib
  ```shell
  pip3 install matplotlib
  ```
- not have numpy
  ```shell
  pip3 install numpy
  ```
