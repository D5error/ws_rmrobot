;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Arm_Joint_Speed_Max)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Arm_Joint_Speed_Max (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::ARM_JOINT_SPEED_MAX")
  (make-package "RM_MSGS::ARM_JOINT_SPEED_MAX"))

(in-package "ROS")
;;//! \htmlinclude Arm_Joint_Speed_Max.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass rm_msgs::Arm_Joint_Speed_Max
  :super ros::object
  :slots (_header _name _max_velocity ))

(defmethod rm_msgs::Arm_Joint_Speed_Max
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:name __name) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:max_velocity __max_velocity) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _name __name)
   (setq _max_velocity __max_velocity)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:max_velocity
   (&optional __max_velocity)
   (if __max_velocity (setq _max_velocity __max_velocity)) _max_velocity)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string[] _name
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _name)) 4
    ;; float64[] _max_velocity
    (* 8    (length _max_velocity)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string[] _name
     (write-long (length _name) s)
     (dolist (elem _name)
       (write-long (length elem) s) (princ elem s)
       )
     ;; float64[] _max_velocity
     (write-long (length _max_velocity) s)
     (dotimes (i (length _max_velocity))
       (sys::poke (elt _max_velocity i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string[] _name
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _name (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _name i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; float64[] _max_velocity
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _max_velocity (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _max_velocity i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get rm_msgs::Arm_Joint_Speed_Max :md5sum-) "96cf1fc11ba4ce2d4aca54cbea5f91a0")
(setf (get rm_msgs::Arm_Joint_Speed_Max :datatype-) "rm_msgs/Arm_Joint_Speed_Max")
(setf (get rm_msgs::Arm_Joint_Speed_Max :definition-)
      "std_msgs/Header header
string[] name
float64[] max_velocity

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :rm_msgs/Arm_Joint_Speed_Max "96cf1fc11ba4ce2d4aca54cbea5f91a0")


