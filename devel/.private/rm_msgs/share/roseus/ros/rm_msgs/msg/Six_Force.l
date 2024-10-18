;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Six_Force)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Six_Force (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::SIX_FORCE")
  (make-package "RM_MSGS::SIX_FORCE"))

(in-package "ROS")
;;//! \htmlinclude Six_Force.msg.html


(defclass rm_msgs::Six_Force
  :super ros::object
  :slots (_force_Fx _force_Fy _force_Fz _force_Mx _force_My _force_Mz ))

(defmethod rm_msgs::Six_Force
  (:init
   (&key
    ((:force_Fx __force_Fx) 0.0)
    ((:force_Fy __force_Fy) 0.0)
    ((:force_Fz __force_Fz) 0.0)
    ((:force_Mx __force_Mx) 0.0)
    ((:force_My __force_My) 0.0)
    ((:force_Mz __force_Mz) 0.0)
    )
   (send-super :init)
   (setq _force_Fx (float __force_Fx))
   (setq _force_Fy (float __force_Fy))
   (setq _force_Fz (float __force_Fz))
   (setq _force_Mx (float __force_Mx))
   (setq _force_My (float __force_My))
   (setq _force_Mz (float __force_Mz))
   self)
  (:force_Fx
   (&optional __force_Fx)
   (if __force_Fx (setq _force_Fx __force_Fx)) _force_Fx)
  (:force_Fy
   (&optional __force_Fy)
   (if __force_Fy (setq _force_Fy __force_Fy)) _force_Fy)
  (:force_Fz
   (&optional __force_Fz)
   (if __force_Fz (setq _force_Fz __force_Fz)) _force_Fz)
  (:force_Mx
   (&optional __force_Mx)
   (if __force_Mx (setq _force_Mx __force_Mx)) _force_Mx)
  (:force_My
   (&optional __force_My)
   (if __force_My (setq _force_My __force_My)) _force_My)
  (:force_Mz
   (&optional __force_Mz)
   (if __force_Mz (setq _force_Mz __force_Mz)) _force_Mz)
  (:serialization-length
   ()
   (+
    ;; float32 _force_Fx
    4
    ;; float32 _force_Fy
    4
    ;; float32 _force_Fz
    4
    ;; float32 _force_Mx
    4
    ;; float32 _force_My
    4
    ;; float32 _force_Mz
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _force_Fx
       (sys::poke _force_Fx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _force_Fy
       (sys::poke _force_Fy (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _force_Fz
       (sys::poke _force_Fz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _force_Mx
       (sys::poke _force_Mx (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _force_My
       (sys::poke _force_My (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _force_Mz
       (sys::poke _force_Mz (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _force_Fx
     (setq _force_Fx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _force_Fy
     (setq _force_Fy (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _force_Fz
     (setq _force_Fz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _force_Mx
     (setq _force_Mx (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _force_My
     (setq _force_My (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _force_Mz
     (setq _force_Mz (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get rm_msgs::Six_Force :md5sum-) "abfa542f676ea571474ea027ddb54a05")
(setf (get rm_msgs::Six_Force :datatype-) "rm_msgs/Six_Force")
(setf (get rm_msgs::Six_Force :definition-)
      "#float32[6] force
float32 force_Fx
float32 force_Fy
float32 force_Fz
float32 force_Mx
float32 force_My
float32 force_Mz

")



(provide :rm_msgs/Six_Force "abfa542f676ea571474ea027ddb54a05")


