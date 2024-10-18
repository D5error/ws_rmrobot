; Auto-generated. Do not edit!


(cl:in-package rm_msgs-msg)


;//! \htmlinclude Six_Force.msg.html

(cl:defclass <Six_Force> (roslisp-msg-protocol:ros-message)
  ((force_Fx
    :reader force_Fx
    :initarg :force_Fx
    :type cl:float
    :initform 0.0)
   (force_Fy
    :reader force_Fy
    :initarg :force_Fy
    :type cl:float
    :initform 0.0)
   (force_Fz
    :reader force_Fz
    :initarg :force_Fz
    :type cl:float
    :initform 0.0)
   (force_Mx
    :reader force_Mx
    :initarg :force_Mx
    :type cl:float
    :initform 0.0)
   (force_My
    :reader force_My
    :initarg :force_My
    :type cl:float
    :initform 0.0)
   (force_Mz
    :reader force_Mz
    :initarg :force_Mz
    :type cl:float
    :initform 0.0))
)

(cl:defclass Six_Force (<Six_Force>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Six_Force>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Six_Force)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rm_msgs-msg:<Six_Force> is deprecated: use rm_msgs-msg:Six_Force instead.")))

(cl:ensure-generic-function 'force_Fx-val :lambda-list '(m))
(cl:defmethod force_Fx-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_Fx-val is deprecated.  Use rm_msgs-msg:force_Fx instead.")
  (force_Fx m))

(cl:ensure-generic-function 'force_Fy-val :lambda-list '(m))
(cl:defmethod force_Fy-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_Fy-val is deprecated.  Use rm_msgs-msg:force_Fy instead.")
  (force_Fy m))

(cl:ensure-generic-function 'force_Fz-val :lambda-list '(m))
(cl:defmethod force_Fz-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_Fz-val is deprecated.  Use rm_msgs-msg:force_Fz instead.")
  (force_Fz m))

(cl:ensure-generic-function 'force_Mx-val :lambda-list '(m))
(cl:defmethod force_Mx-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_Mx-val is deprecated.  Use rm_msgs-msg:force_Mx instead.")
  (force_Mx m))

(cl:ensure-generic-function 'force_My-val :lambda-list '(m))
(cl:defmethod force_My-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_My-val is deprecated.  Use rm_msgs-msg:force_My instead.")
  (force_My m))

(cl:ensure-generic-function 'force_Mz-val :lambda-list '(m))
(cl:defmethod force_Mz-val ((m <Six_Force>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:force_Mz-val is deprecated.  Use rm_msgs-msg:force_Mz instead.")
  (force_Mz m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Six_Force>) ostream)
  "Serializes a message object of type '<Six_Force>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_Fx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_Fy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_Fz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_Mx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_My))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'force_Mz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Six_Force>) istream)
  "Deserializes a message object of type '<Six_Force>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_Fx) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_Fy) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_Fz) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_Mx) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_My) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'force_Mz) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Six_Force>)))
  "Returns string type for a message object of type '<Six_Force>"
  "rm_msgs/Six_Force")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Six_Force)))
  "Returns string type for a message object of type 'Six_Force"
  "rm_msgs/Six_Force")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Six_Force>)))
  "Returns md5sum for a message object of type '<Six_Force>"
  "abfa542f676ea571474ea027ddb54a05")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Six_Force)))
  "Returns md5sum for a message object of type 'Six_Force"
  "abfa542f676ea571474ea027ddb54a05")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Six_Force>)))
  "Returns full string definition for message of type '<Six_Force>"
  (cl:format cl:nil "#float32[6] force~%float32 force_Fx~%float32 force_Fy~%float32 force_Fz~%float32 force_Mx~%float32 force_My~%float32 force_Mz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Six_Force)))
  "Returns full string definition for message of type 'Six_Force"
  (cl:format cl:nil "#float32[6] force~%float32 force_Fx~%float32 force_Fy~%float32 force_Fz~%float32 force_Mx~%float32 force_My~%float32 force_Mz~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Six_Force>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Six_Force>))
  "Converts a ROS message object to a list"
  (cl:list 'Six_Force
    (cl:cons ':force_Fx (force_Fx msg))
    (cl:cons ':force_Fy (force_Fy msg))
    (cl:cons ':force_Fz (force_Fz msg))
    (cl:cons ':force_Mx (force_Mx msg))
    (cl:cons ':force_My (force_My msg))
    (cl:cons ':force_Mz (force_Mz msg))
))
