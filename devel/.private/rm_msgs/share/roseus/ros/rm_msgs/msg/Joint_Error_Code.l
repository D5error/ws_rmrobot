;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Joint_Error_Code)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Joint_Error_Code (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::JOINT_ERROR_CODE")
  (make-package "RM_MSGS::JOINT_ERROR_CODE"))

(in-package "ROS")
;;//! \htmlinclude Joint_Error_Code.msg.html


(defclass rm_msgs::Joint_Error_Code
  :super ros::object
  :slots (_joint_error ))

(defmethod rm_msgs::Joint_Error_Code
  (:init
   (&key
    ((:joint_error __joint_error) (make-array 6 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _joint_error __joint_error)
   self)
  (:joint_error
   (&optional __joint_error)
   (if __joint_error (setq _joint_error __joint_error)) _joint_error)
  (:serialization-length
   ()
   (+
    ;; uint16[6] _joint_error
    (* 2    6)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16[6] _joint_error
     (dotimes (i 6)
       (write-word (elt _joint_error i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16[6] _joint_error
   (dotimes (i (length _joint_error))
     (setf (elt _joint_error i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;;
   self)
  )

(setf (get rm_msgs::Joint_Error_Code :md5sum-) "74ddce861d3ff625b60dae7918fad457")
(setf (get rm_msgs::Joint_Error_Code :datatype-) "rm_msgs/Joint_Error_Code")
(setf (get rm_msgs::Joint_Error_Code :definition-)
      "uint16[6] joint_error   #每个关节报错信息

")



(provide :rm_msgs/Joint_Error_Code "74ddce861d3ff625b60dae7918fad457")


