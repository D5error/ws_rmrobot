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

class Joint_Max_Speed {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.joint_num = null;
      this.joint_max_speed = null;
    }
    else {
      if (initObj.hasOwnProperty('joint_num')) {
        this.joint_num = initObj.joint_num
      }
      else {
        this.joint_num = 0;
      }
      if (initObj.hasOwnProperty('joint_max_speed')) {
        this.joint_max_speed = initObj.joint_max_speed
      }
      else {
        this.joint_max_speed = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Joint_Max_Speed
    // Serialize message field [joint_num]
    bufferOffset = _serializer.uint8(obj.joint_num, buffer, bufferOffset);
    // Serialize message field [joint_max_speed]
    bufferOffset = _serializer.float32(obj.joint_max_speed, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Joint_Max_Speed
    let len;
    let data = new Joint_Max_Speed(null);
    // Deserialize message field [joint_num]
    data.joint_num = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [joint_max_speed]
    data.joint_max_speed = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 5;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rm_msgs/Joint_Max_Speed';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9a15b693ccbb220eba8aa0b693b24585';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 joint_num
    float32 joint_max_speed
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Joint_Max_Speed(null);
    if (msg.joint_num !== undefined) {
      resolved.joint_num = msg.joint_num;
    }
    else {
      resolved.joint_num = 0
    }

    if (msg.joint_max_speed !== undefined) {
      resolved.joint_max_speed = msg.joint_max_speed;
    }
    else {
      resolved.joint_max_speed = 0.0
    }

    return resolved;
    }
};

module.exports = Joint_Max_Speed;
