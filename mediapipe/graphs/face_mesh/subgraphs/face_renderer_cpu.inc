"\n?\022\031ImagePropertiesCalculator\032\021IMAGE:input_image\"\017SIZE:image_"
"size\n\272\001\022 DetectionsToRenderDataCalculator\032\025DETECTIONS:detectio"
"ns\"\"RENDER_DATA:detections_render_dataB[\nEtype.googleapis.com/mediapipe.Det"
"ectionsToRenderDataCalculatorOptions\022\022)\000\000\000\000\000\000\020@2"
"\007\010\000\020\377\001\030\000\n\203\001\022/BeginLoopNormalizedLandmarkList"
"VectorCalculator\032\035ITERABLE:multi_face_landmarks\"\023ITEM:face_landmarks"
"\"\034BATCH_END:landmark_timestamp\n\316\001\022#FaceLandmarksToRenderDataCalc"
"ulator\032\035NORM_LANDMARKS:face_landmarks\"!RENDER_DATA:landmarks_render_dat"
"aBe\nDtype.googleapis.com/mediapipe.LandmarksToRenderDataCalculatorOptions\022"
"\035\022\007\010\377\001\020\000\030\000\032\007\010\000\020\377\001\030\000!"
"\000\000\000\000\000\000\000@(\000\n\202\001\022\033EndLoopRenderDataCalculato"
"r\032\032ITEM:landmarks_render_data\032\034BATCH_END:landmark_timestamp\")ITER"
"ABLE:multi_face_landmarks_render_data\n\246\001\022\032RectToRenderDataCalcula"
"tor\032\020NORM_RECTS:rects\"\035RENDER_DATA:rects_render_dataBW\n?type.google"
"apis.com/mediapipe.RectToRenderDataCalculatorOptions\022\024\010\000\022\007"
"\010\377\001\020\000\030\000\031\000\000\000\000\000\000\020@\n\232\001\022"
"\033AnnotationOverlayCalculator\032\021IMAGE:input_image\032\026detections_ren"
"der_data\032)VECTOR:0:multi_face_landmarks_render_data\032\021rects_render_dat"
"a\"\022IMAGE:output_imageR\021IMAGE:input_imageR\036LANDMARKS:multi_face_landm"
"arksR\020NORM_RECTS:rectsR\025DETECTIONS:detectionsz\022IMAGE:output_image\242"
"\001\017FaceRendererCpu"