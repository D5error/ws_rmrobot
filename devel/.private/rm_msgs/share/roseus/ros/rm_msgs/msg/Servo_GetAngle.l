;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Servo_GetAngle)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Servo_GetAngle (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::SERVO_GETANGLE")
  (make-package "RM_MSGS::SERVO_GETANGLE"))

(in-package "ROS")
;;//! \htmlinclude Servo_GetAngle.msg.html


(defclass rm_msgs::Servo_GetAngle
  :super ros::object
  :slots (_servo_id ))

(defmethod rm_msgs::Servo_GetAngle
  (:init
   (&key
    ((:servo_id __servo_id) 0)
    )
   (send-super :init)
   (setq _servo_id (round __servo_id))
   self)
  (:servo_id
   (&optional __servo_id)
   (if __servo_id (setq _servo_id __servo_id)) _servo_id)
  (:serialization-length
   ()
   (+
    ;; uint16 _servo_id
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16 _servo_id
       (write-word _servo_id s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16 _servo_id
     (setq _servo_id (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get rm_msgs::Servo_GetAngle :md5sum-) "44a63dfe689e2d7241879f9eb59bd488")
(setf (get rm_msgs::Servo_GetAngle :datatype-) "rm_msgs/Servo_GetAngle")
(setf (get rm_msgs::Servo_GetAngle :definition-)
      "#读取舵机角度位置值
uint16 servo_id	#舵机ID

")



(provide :rm_msgs/Servo_GetAngle "44a63dfe689e2d7241879f9eb59bd488")


