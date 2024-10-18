;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Set_Force_Position)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Set_Force_Position (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::SET_FORCE_POSITION")
  (make-package "RM_MSGS::SET_FORCE_POSITION"))

(in-package "ROS")
;;//! \htmlinclude Set_Force_Position.msg.html


(defclass rm_msgs::Set_Force_Position
  :super ros::object
  :slots (_sensor _mode _direction _N ))

(defmethod rm_msgs::Set_Force_Position
  (:init
   (&key
    ((:sensor __sensor) 0)
    ((:mode __mode) 0)
    ((:direction __direction) 0)
    ((:N __N) 0)
    )
   (send-super :init)
   (setq _sensor (round __sensor))
   (setq _mode (round __mode))
   (setq _direction (round __direction))
   (setq _N (round __N))
   self)
  (:sensor
   (&optional __sensor)
   (if __sensor (setq _sensor __sensor)) _sensor)
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:direction
   (&optional __direction)
   (if __direction (setq _direction __direction)) _direction)
  (:N
   (&optional __N)
   (if __N (setq _N __N)) _N)
  (:serialization-length
   ()
   (+
    ;; uint8 _sensor
    1
    ;; uint8 _mode
    1
    ;; uint8 _direction
    1
    ;; int16 _N
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _sensor
       (write-byte _sensor s)
     ;; uint8 _mode
       (write-byte _mode s)
     ;; uint8 _direction
       (write-byte _direction s)
     ;; int16 _N
       (write-word _N s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _sensor
     (setq _sensor (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _mode
     (setq _mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _direction
     (setq _direction (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int16 _N
     (setq _N (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get rm_msgs::Set_Force_Position :md5sum-) "8d9c11c3bbe640daba4f629758cfa057")
(setf (get rm_msgs::Set_Force_Position :datatype-) "rm_msgs/Set_Force_Position")
(setf (get rm_msgs::Set_Force_Position :definition-)
      "uint8 sensor
uint8 mode
uint8 direction
int16 N
")



(provide :rm_msgs/Set_Force_Position "8d9c11c3bbe640daba4f629758cfa057")


