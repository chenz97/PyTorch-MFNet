import os
import logging
import cv2
# import subprocess
#
# from joblib import delayed
# from joblib import Parallel


def convert_video_wapper(src_videos, 
                         dst_videos):
    for src, dst in zip(src_videos, dst_videos):
        print('dealing with ', src)
        vidcap = cv2.VideoCapture(src)
        cnt = 0
        success, image = vidcap.read()
        while success:
            cv2.imwrite(os.path.join(dst, '%05d.jpg' % cnt), image)
            success, image = vidcap.read()
            cnt += 1
        print(' - Done. Processed {} frames'.format(cnt))

def count_frames(vid_path, check_validity=True):
    offset = 0
    cap = cv2.VideoCapture(vid_path)
    if vid_path.endswith('.flv'):
        offset = -1
    unverified_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
    if check_validity:
        verified_frame_count = 0
        for i in range(unverified_frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            if not cap.grab():
                logging.warning("VideoIter:: >> frame (start from 0) {} corrupted in {}".format(i, vid_path))
                break
            verified_frame_count = i + 1
        frame_count = verified_frame_count
    else:
        frame_count = unverified_frame_count
    return frame_count

def check(src_videos,
          dst_videos):
    for src, dst in zip(src_videos, dst_videos):
        cnt1 = count_frames(src, False)
        cnt2 = len(os.listdir(dst))
        if cnt1 - cnt2 > 1:
            print(src, cnt1, cnt2)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    # resize to slen = x360
    # cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an {}'
    # cmd_format = 'ffmpeg -i {} -q:v 2 {}/%05d.jpg'

    src_root = '../raw/data-x360'
    dst_root = '../raw/frames'
    assert os.path.exists(dst_root), "cannot locate `{}'".format(dst_root)

    classname = [name for name in os.listdir(src_root) \
                    if os.path.isdir(os.path.join(src_root,name))]
    classname.sort()

    for cls_name in classname:
        src_folder = os.path.join(src_root, cls_name)
        dst_folder = os.path.join(dst_root, cls_name)
        assert os.path.exists(src_folder), "failed to locate: `{}'.".format(src_folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        video_names = [name for name in os.listdir(src_folder) \
                            if os.path.isfile(os.path.join(src_folder, name))]

        for vid_name in video_names:
            folder = os.path.join(dst_folder, vid_name[:-4])
            if not os.path.exists(folder):
                os.makedirs(folder)

        # src_videos = [os.path.join(src_folder, vid_name.replace(";", "\;").replace("&", "\&")) for vid_name in video_names]
        # dst_videos = [os.path.join(dst_folder, vid_name.replace(";", "\;").replace("&", "\&")) for vid_name in video_names]

        src_videos = [os.path.join(src_folder, vid_name) for vid_name in video_names]
        dst_videos = [os.path.join(dst_folder, vid_name[:-4]) for vid_name in video_names]

        convert_video_wapper(src_videos=src_videos,
                             dst_videos=dst_videos)
        # check(src_videos, dst_videos)
    # logging.info("- Done.")
