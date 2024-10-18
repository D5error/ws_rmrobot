; Auto-generated. Do not edit!


(cl:in-package rm_msgs-msg)


;//! \htmlinclude Joint_Max_Speed.msg.html

(cl:defclass <Joint_Max_Speed> (roslisp-msg-protocol:ros-message)
  ((joint_num
    :reader joint_num
    :initarg :joint_num
    :type cl:fixnum
    :initform 0)
   (joint_max_speed
    :reader joint_max_speed
    :initarg :joint_max_speed
    :type cl:float
    :initform 0.0))
)

(cl:defclass Joint_Max_Speed (<Joint_Max_Speed>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Joint_Max_Speed>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Joint_Max_Speed)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rm_msgs-msg:<Joint_Max_Speed> is deprecated: use rm_msgs-msg:Joint_Max_Speed instead.")))

(cl:ensure-generic-function 'joint_num-val :lambda-list '(m))
(cl:defmethod joint_num-val ((m <Joint_Max_Speed>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:joint_num-val is deprecated.  Use rm_msgs-msg:joint_num instead.")
  (joint_num m))

(cl:ensure-generic-function 'joint_max_speed-val :lambda-list '(m))
(cl:defmethod joint_max_speed-val ((m <Joint_Max_Speed>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:joint_max_speed-val is deprecated.  Use rm_msgs-msg:joint_max_speed instead.")
  (joint_max_speed m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Joint_Max_Speed>) ostream)
  "Serializes a message object of type '<Joint_Max_Speed>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'joint_num)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'joint_max_speed))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Joint_Max_Speed>) istream)
  "Deserializes a message object of type '<Joint_Max_Speed>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'joint_num)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'joint_max_speed) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Joint_Max_Speed>)))
  "Returns string type for a message object of type '<Joint_Max_Speed>"
  "rm_msgs/Joint_Max_Speed")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Joint_Max_Speed)))
  "Returns string type for a message object of type 'Joint_Max_Speed"
  "rm_msgs/Joint_Max_Speed")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Joint_Max_Speed>)))
  "Returns md5sum for a message object of type '<Joint_Max_Speed>"
  "9a15b693ccbb220eba8aa0b693b24585")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Joint_Max_Speed)))
  "Returns md5sum for a message object of type 'Joint_Max_Speed"
  "9a15b693ccbb220eba8aa0b693b24585")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Joint_Max_Speed>)))
  "Returns full string definition for message of type '<Joint_Max_Speed>"
  (cl:format cl:nil "uint8 joint_num~%float32 joint_max_speed~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Joint_Max_Speed)))
  "Returns full string definition for message of type 'Joint_Max_Speed"
  (cl:format cl:nil "uint8 joint_num~%float32 joint_max_speed~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Joint_Max_Speed>))
  (cl:+ 0
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Joint_Max_Speed>))
  "Converts a ROS message object to a list"
  (cl:list 'Joint_Max_Speed
    (cl:cons ':joint_num (joint_num msg))
    (cl:cons ':joint_max_speed (joint_max_speed msg))
))
