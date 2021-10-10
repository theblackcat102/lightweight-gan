
sudo docker run --gpus all -it  \
    --shm-size=10gb \
    -v /home/theblackcat102/img_corpus:/data/ \
    -v /mnt/ssd1/:/ssd1/ \
    -v /home/theblackcat102/Project/lightweight-gan:/workspace/lightweight-gan \
    nvcr.io/nvidia/pytorch:21.03-py3