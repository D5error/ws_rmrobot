;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Set_Realtime_Push)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Set_Realtime_Push (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::SET_REALTIME_PUSH")
  (make-package "RM_MSGS::SET_REALTIME_PUSH"))

(in-package "ROS")
;;//! \htmlinclude Set_Realtime_Push.msg.html


(defclass rm_msgs::Set_Realtime_Push
  :super ros::object
  :slots (_cycle _port ))

(defmethod rm_msgs::Set_Realtime_Push
  (:init
   (&key
    ((:cycle __cycle) 0)
    ((:port __port) 0)
    )
   (send-super :init)
   (setq _cycle (round __cycle))
   (setq _port (round __port))
   self)
  (:cycle
   (&optional __cycle)
   (if __cycle (setq _cycle __cycle)) _cycle)
  (:port
   (&optional __port)
   (if __port (setq _port __port)) _port)
  (:serialization-length
   ()
   (+
    ;; uint16 _cycle
    2
    ;; uint16 _port
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16 _cycle
       (write-word _cycle s)
     ;; uint16 _port
       (write-word _port s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16 _cycle
     (setq _cycle (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _port
     (setq _port (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get rm_msgs::Set_Realtime_Push :md5sum-) "27a166430262b6d68578edf0f7b5398f")
(setf (get rm_msgs::Set_Realtime_Push :datatype-) "rm_msgs/Set_Realtime_Push")
(setf (get rm_msgs::Set_Realtime_Push :definition-)
      "uint16 cycle
uint16 port
")



(provide :rm_msgs/Set_Realtime_Push "27a166430262b6d68578edf0f7b5398f")


