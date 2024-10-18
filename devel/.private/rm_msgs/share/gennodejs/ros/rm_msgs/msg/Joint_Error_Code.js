// Auto-generated. Do not edit!

// (in-package rm_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Joint_Error_Code {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.joint_error = null;
    }
    else {
      if (initObj.hasOwnProperty('joint_error')) {
        this.joint_error = initObj.joint_error
      }
      else {
        this.joint_error = new Array(6).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Joint_Error_Code
    // Check that the constant length array field [joint_error] has the right length
    if (obj.joint_error.length !== 6) {
      throw new Error('Unable to serialize array field joint_error - length must be 6')
    }
    // Serialize message field [joint_error]
    bufferOffset = _arraySerializer.uint16(obj.joint_error, buffer, bufferOffset, 6);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Joint_Error_Code
    let len;
    let data = new Joint_Error_Code(null);
    // Deserialize message field [joint_error]
    data.joint_error = _arrayDeserializer.uint16(buffer, bufferOffset, 6)
    return data;
  }

  static getMessageSize(object) {
    return 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rm_msgs/Joint_Error_Code';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '74ddce861d3ff625b60dae7918fad457';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint16[6] joint_error   #每个关节报错信息
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Joint_Error_Code(null);
    if (msg.joint_error !== undefined) {
      resolved.joint_error = msg.joint_error;
    }
    else {
      resolved.joint_error = new Array(6).fill(0)
    }

    return resolved;
    }
};

module.exports = Joint_Error_Code;
