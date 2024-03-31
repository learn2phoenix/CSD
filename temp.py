# python code to dump frames of multiple video files in parallel using ffmpeg
#
import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# ffmpeg command to dump video frames at original fps which is not 10


# def dump_frames(video_file, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     subprocess.call(['ffmpeg', '-i', video_file, os.path.join(output_dir, '%05d.jpg')])
#
# def main():
#     video_dir = '/fs/cfar-projects/VPR/FVD/train'
#     output_dir = '/fs/cfar-projects/VPR/FVD/train_images'
#     os.makedirs(output_dir, exist_ok=True)
#     videos = glob.glob(video_dir + '/**/*.mp4')
#     with ProcessPoolExecutor(max_workers=15) as executor:
#         futures = [executor.submit(dump_frames, video, os.path.join(output_dir, video.split('/')[-1].split('.')[0])) for video in videos]
#         for future in as_completed(futures):
#             print(future.result())

def main():
    video_dir = '/fs/cfar-projects/VPR/FVD/train'
    output_dir = '/fs/cfar-projects/VPR/FVD/train_images_new'

    for idx, video in enumerate(os.listdir(output_dir)):
        os.rename(os.path.join(output_dir, video), os.path.join(output_dir, f'{str(idx).zfill(3)}'))

if __name__ == '__main__':
    main()




