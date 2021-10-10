import os, glob
from torchvision.io.image import read_image
from tqdm import tqdm
'''
Check image of a directory

'''


if __name__ == "__main__":
    parsed = []
    success_file = 'success_cc12M.txt'
    if not os.path.exists(success_file):
        with open(success_file, 'w') as f:
            f.write('')
    else:
        with open(success_file, 'r') as f:
            parsed = f.read().split('\n')
        parsed = set(parsed)

    for filename in tqdm(glob.glob('/home/theblackcat102/img_corpus/unsplash_caption/*')):
        if os.path.splitext(filename) in ['.jpg', '.png'] and filename not in parsed:
            try:
                read_image(filename)
                with open(success_file, 'a') as f:
                    f.write(filename+'\n')
            except Exception as e:
                print(e, filename)
