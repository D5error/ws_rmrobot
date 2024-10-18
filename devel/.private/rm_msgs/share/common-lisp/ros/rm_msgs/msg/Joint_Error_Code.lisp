; Auto-generated. Do not edit!


(cl:in-package rm_msgs-msg)


;//! \htmlinclude Joint_Error_Code.msg.html

(cl:defclass <Joint_Error_Code> (roslisp-msg-protocol:ros-message)
  ((joint_error
    :reader joint_error
    :initarg :joint_error
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 6 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass Joint_Error_Code (<Joint_Error_Code>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Joint_Error_Code>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Joint_Error_Code)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rm_msgs-msg:<Joint_Error_Code> is deprecated: use rm_msgs-msg:Joint_Error_Code instead.")))

(cl:ensure-generic-function 'joint_error-val :lambda-list '(m))
(cl:defmethod joint_error-val ((m <Joint_Error_Code>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rm_msgs-msg:joint_error-val is deprecated.  Use rm_msgs-msg:joint_error instead.")
  (joint_error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Joint_Error_Code>) ostream)
  "Serializes a message object of type '<Joint_Error_Code>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'joint_error))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Joint_Error_Code>) istream)
  "Deserializes a message object of type '<Joint_Error_Code>"
  (cl:setf (cl:slot-value msg 'joint_error) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'joint_error)))
    (cl:dotimes (i 6)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Joint_Error_Code>)))
  "Returns string type for a message object of type '<Joint_Error_Code>"
  "rm_msgs/Joint_Error_Code")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Joint_Error_Code)))
  "Returns string type for a message object of type 'Joint_Error_Code"
  "rm_msgs/Joint_Error_Code")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Joint_Error_Code>)))
  "Returns md5sum for a message object of type '<Joint_Error_Code>"
  "74ddce861d3ff625b60dae7918fad457")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Joint_Error_Code)))
  "Returns md5sum for a message object of type 'Joint_Error_Code"
  "74ddce861d3ff625b60dae7918fad457")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Joint_Error_Code>)))
  "Returns full string definition for message of type '<Joint_Error_Code>"
  (cl:format cl:nil "uint16[6] joint_error   #每个关节报错信息~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Joint_Error_Code)))
  "Returns full string definition for message of type 'Joint_Error_Code"
  (cl:format cl:nil "uint16[6] joint_error   #每个关节报错信息~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Joint_Error_Code>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_error) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Joint_Error_Code>))
  "Converts a ROS message object to a list"
  (cl:list 'Joint_Error_Code
    (cl:cons ':joint_error (joint_error msg))
))
