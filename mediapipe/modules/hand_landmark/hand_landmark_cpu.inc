"\n\222\001\022\027ImageToTensorCalculator\032\013IMAGE:image\032\023NORM_RECT:"
"hand_rect\"\024TENSORS:input_tensor\"#LETTERBOX_PADDING:letterbox_padding:\032"
"\232\325\276\373\t\024\010\340\001\020\340\001\030\001\"\n\r\000\000\000\000"
"\025\000\000\200?\nI\022\027HandLandmarkModelLoader*!MODEL_COMPLEXITY:model_co"
"mplexity2\013MODEL:model\n\\\022\023InferenceCalculator\032\024TENSORS:input_t"
"ensor\"\026TENSORS:output_tensors*\013MODEL:model:\n\272\237\335\204\n\004*"
"\002\"\000\n\233\001\022\033SplitTensorVectorCalculator\032\016output_tensors"
"\"\020landmark_tensors\"\020hand_flag_tensor\"\021handedness_tensor\"\025world"
"_landmark_tensor:\036\362\350\326\335\007\030\n\004\010\000\020\001\n\004\010"
"\001\020\002\n\004\010\002\020\003\n\004\010\003\020\004\nP\022\031TensorsToFl"
"oatsCalculator\032\030TENSORS:hand_flag_tensor\"\031FLOAT:hand_presence_score"
"\nX\022\026ThresholdingCalculator\032\031FLOAT:hand_presence_score\"\022FLAG:h"
"and_presence:\017\222\276\344\337\007\t\t\000\000\000\000\000\000\340?\nS\022"
"\016GateCalculator\032\021handedness_tensor\032\023ALLOW:hand_presence\"\031en"
"sured_handedness_tensor\n\236\001\022!TensorsToClassificationCalculator\032!TE"
"NSORS:ensured_handedness_tensor\"\032CLASSIFICATIONS:handedness::\362\352\340"
"\200\n4\020\001\032.mediapipe/modules/hand_landmark/handedness.txt \001\nQ\022"
"\016GateCalculator\032\020landmark_tensors\032\023ALLOW:hand_presence\"\030ens"
"ured_landmark_tensors\no\022\034TensorsToLandmarksCalculator\032 TENSORS:ensur"
"ed_landmark_tensors\"\030NORM_LANDMARKS:landmarks:\023\202\353\340\200\n\r\010"
"\025\020\340\001\030\340\001-\315\314\314>\nz\022\"LandmarkLetterboxRemovalCal"
"culator\032\023LANDMARKS:landmarks\032#LETTERBOX_PADDING:letterbox_padding\""
"\032LANDMARKS:scaled_landmarks\ns\022\034LandmarkProjectionCalculator\032\037N"
"ORM_LANDMARKS:scaled_landmarks\032\023NORM_RECT:hand_rect\"\035NORM_LANDMARKS:"
"hand_landmarks\n[\022\016GateCalculator\032\025world_landmark_tensor\032\023AL"
"LOW:hand_presence\"\035ensured_world_landmark_tensor\nv\022\034TensorsToLandma"
"rksCalculator\032%TENSORS:ensured_world_landmark_tensor\"%LANDMARKS:unprojecte"
"d_world_landmarks:\010\202\353\340\200\n\002\010\025\n\177\022!WorldLandmarkPr"
"ojectionCalculator\032%LANDMARKS:unprojected_world_landmarks\032\023NORM_RECT:"
"hand_rect\"\036LANDMARKS:hand_world_landmarksR\013IMAGE:imageR\rROI:hand_rectz"
"\030LANDMARKS:hand_landmarksz$WORLD_LANDMARKS:hand_world_landmarksz\025HANDEDN"
"ESS:handedness\202\001!MODEL_COMPLEXITY:model_complexity\242\001\017HandLandma"
"rkCpu"