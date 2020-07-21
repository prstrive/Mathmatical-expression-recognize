# Mathmatical-expression-recognize
Convert the printed mathematical expression in the picture to latex by tensorflow.

### Start
1. Train the symbol classify CNN from `symbol_classer_cnn.py`.
2. Choose the best classer and assign it's path to variable `symClasser` in `global_variable.py`
3. Run the interface file `main.py`. 
### Execution procedure of `main.py`
- translate RGB image to binary image.
- get the segmentations of the binary image.
- recognise the symbol of each segmentation.
- structure analyse.
- get latex code.
