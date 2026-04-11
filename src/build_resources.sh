echo "In order to run this tool, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

mkdir -p assets
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/FLAME_with_eye.pt -O ./assets/FLAME_with_eye.pt
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/GAGAvatar.pt -O ./assets/GAGAvatar/GAGAvatar.pt

wget https://huggingface.co/xg-chu/ARTalk/resolve/main/ARTalk_wav2vec.pt -O ./assets/ARTalk_wav2vec.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/config.json -O ./assets/config.json
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/GAGAvatar/tracked.pt -O ./assets/GAGAvatar/tracked.pt

wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/angry_0.pt -O ./assets/style_motion/angry_0.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/curious_0.pt -O ./assets/style_motion/curious_0.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/doubtful_0.pt -O ./assets/style_motion/doubtful_0.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/doubtful_1.pt -O ./assets/style_motion/doubtful_1.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/happy_0.pt -O ./assets/style_motion/happy_0.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/happy_1.pt -O ./assets/style_motion/happy_1.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/happy_2.pt -O ./assets/style_motion/happy_2.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_0.pt -O ./assets/style_motion/natural_0.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_1.pt -O ./assets/style_motion/natural_1.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_2.pt -O ./assets/style_motion/natural_2.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_3.pt -O ./assets/style_motion/natural_3.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_4.pt -O ./assets/style_motion/natural_4.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_5.pt -O ./assets/style_motion/natural_5.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_6.pt -O ./assets/style_motion/natural_6.pt
wget https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_7.pt -O ./assets/style_motion/natural_7.pt
