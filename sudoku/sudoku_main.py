
from number_recognition import Number_recognition
from sudoku_cv import Sudoku_CV
if __name__ == '__main__':
    test_images_str =  'train-images.idx3-ubyte'
    test_labels_str =  'train-labels.idx1-ubyte'

    nr = Number_recognition ( test_images_str, test_labels_str, 10);

    nr.load_train()
        
    sudoku = Sudoku_CV('sudoku.jpg');
    sudoku.findRectangle_Sudoku()
    sudoku.Recognize_number(1, 0, nr)

    sudoku.Recognize_sudoku(nr)


