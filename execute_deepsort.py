# Tensorflow Import

# deep sort imports
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Definition of the parameters
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = None
MODEL_FILENAME = 'model_data/mars-small128.pb'
NMS_MAX_OVERLAP = 1.0

# initialize tracker
encoder = gdet.create_box_encoder(MODEL_FILENAME, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric)


def execute_tracker(frame, bboxes, scores, class_names):
    """
    Execute tracker detector 

    Normally, we can get list objects after 2 frames.
    All the processes will be passed to non_max_suppression before going into execute_tracker
    We use Kalman Filter + Cosine distance to update the state of tracker

    Parameters:
    -----------
    frame, bboxes, scores, class_names
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
    # indices = preprocessing.non_max_suppression(boxs, classes, NMS_MAX_OVERLAP, scores) # have use non-max-suppression before
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
