; Auto-generated. Do not edit!


(cl:in-package rm_msgs-msg)


;//! \htmlinclude Set_Realtime_Push.msg.html

(cl:defclass <Set_Realtime_Push> (roslisp-msg-protocol:ros-message)
  ((cycle
    :reader cycle
    :initarg :cycle
    :type cl:fixnum
    :initform 0)
   (port
    :reader port
    :initarg :port
    :type cl:fixnum
    :initform 0))
)

(cl:defclass Set_Realtime_Push (<Set_Realtime_Push>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Set_Realtime_Push>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Set_Realtime_Push)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rm_msgs-msg:<Set_Realtime_Push> is deprecated: use rm_msgs-msg:Set_Realtime_Push instead.")))

(cl:ensure-generic-function 'cycle-val :lambda-list '(m))
(cl:defmethod cycle-val ((m <Set_Realtime_Push>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:cycle-val is deprecated.  Use rm_msgs-msg:cycle instead.")
  (cycle m))

(cl:ensure-generic-function 'port-val :lambda-list '(m))
(cl:defmethod port-val ((m <Set_Realtime_Push>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:port-val is deprecated.  Use rm_msgs-msg:port instead.")
  (port m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Set_Realtime_Push>) ostream)
  "Serializes a message object of type '<Set_Realtime_Push>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'cycle)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'cycle)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'port)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'port)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Set_Realtime_Push>) istream)
  "Deserializes a message object of type '<Set_Realtime_Push>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'cycle)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'cycle)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'port)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'port)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Set_Realtime_Push>)))
  "Returns string type for a message object of type '<Set_Realtime_Push>"
  "rm_msgs/Set_Realtime_Push")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Set_Realtime_Push)))
  "Returns string type for a message object of type 'Set_Realtime_Push"
  "rm_msgs/Set_Realtime_Push")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Set_Realtime_Push>)))
  "Returns md5sum for a message object of type '<Set_Realtime_Push>"
  "27a166430262b6d68578edf0f7b5398f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Set_Realtime_Push)))
  "Returns md5sum for a message object of type 'Set_Realtime_Push"
  "27a166430262b6d68578edf0f7b5398f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Set_Realtime_Push>)))
  "Returns full string definition for message of type '<Set_Realtime_Push>"
  (cl:format cl:nil "uint16 cycle~%uint16 port~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Set_Realtime_Push)))
  "Returns full string definition for message of type 'Set_Realtime_Push"
  (cl:format cl:nil "uint16 cycle~%uint16 port~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Set_Realtime_Push>))
  (cl:+ 0
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Set_Realtime_Push>))
  "Converts a ROS message object to a list"
  (cl:list 'Set_Realtime_Push
    (cl:cons ':cycle (cycle msg))
    (cl:cons ':port (port msg))
))
