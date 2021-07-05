# Tensorflow Import

# deep sort imports
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def execute_tracker(frame, bboxes, scores, class_names):
    """
    Execute tracker detector 

    Normally, we can get list objects after 2 frames.
    All the processes will be passed to non_max_suppression before going into execute_tracker
    We use Kalman Filter + Cosine distance to update the state of tracker

    Parameters:
    -----------
    frame, bboxes, scores, class_names
    bboxes: x_min, y_min, width, height
    Returns:
    -----------
    items_arr
    """
    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(bboxes, scores, class_names, features)]

    # boxs = np.array([d.tlwh for d in detections]) # Get the box locations
    # scores = np.array([d.confidence for d in detections])
    # classes = np.array([d.class_name for d in detections])
    # indices = preprocessing.non_max_suppression(boxs, classes, 0.3, 0) # have use non-max-suppression before
    # detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    # Get all objects with track id
    items_arr = {}
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        # Add all tracking items into array
        items_arr[track.track_id] = {"bbox": bbox, "class_name": class_name}

    return items_arr
