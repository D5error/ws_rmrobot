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

class Six_Force {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.force_Fx = null;
      this.force_Fy = null;
      this.force_Fz = null;
      this.force_Mx = null;
      this.force_My = null;
      this.force_Mz = null;
    }
    else {
      if (initObj.hasOwnProperty('force_Fx')) {
        this.force_Fx = initObj.force_Fx
      }
      else {
        this.force_Fx = 0.0;
      }
      if (initObj.hasOwnProperty('force_Fy')) {
        this.force_Fy = initObj.force_Fy
      }
      else {
        this.force_Fy = 0.0;
      }
      if (initObj.hasOwnProperty('force_Fz')) {
        this.force_Fz = initObj.force_Fz
      }
      else {
        this.force_Fz = 0.0;
      }
      if (initObj.hasOwnProperty('force_Mx')) {
        this.force_Mx = initObj.force_Mx
      }
      else {
        this.force_Mx = 0.0;
      }
      if (initObj.hasOwnProperty('force_My')) {
        this.force_My = initObj.force_My
      }
      else {
        this.force_My = 0.0;
      }
      if (initObj.hasOwnProperty('force_Mz')) {
        this.force_Mz = initObj.force_Mz
      }
      else {
        this.force_Mz = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Six_Force
    // Serialize message field [force_Fx]
    bufferOffset = _serializer.float32(obj.force_Fx, buffer, bufferOffset);
    // Serialize message field [force_Fy]
    bufferOffset = _serializer.float32(obj.force_Fy, buffer, bufferOffset);
    // Serialize message field [force_Fz]
    bufferOffset = _serializer.float32(obj.force_Fz, buffer, bufferOffset);
    // Serialize message field [force_Mx]
    bufferOffset = _serializer.float32(obj.force_Mx, buffer, bufferOffset);
    // Serialize message field [force_My]
    bufferOffset = _serializer.float32(obj.force_My, buffer, bufferOffset);
    // Serialize message field [force_Mz]
    bufferOffset = _serializer.float32(obj.force_Mz, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Six_Force
    let len;
    let data = new Six_Force(null);
    // Deserialize message field [force_Fx]
    data.force_Fx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [force_Fy]
    data.force_Fy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [force_Fz]
    data.force_Fz = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [force_Mx]
    data.force_Mx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [force_My]
    data.force_My = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [force_Mz]
    data.force_Mz = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rm_msgs/Six_Force';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'abfa542f676ea571474ea027ddb54a05';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #float32[6] force
    float32 force_Fx
    float32 force_Fy
    float32 force_Fz
    float32 force_Mx
    float32 force_My
    float32 force_Mz
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Six_Force(null);
    if (msg.force_Fx !== undefined) {
      resolved.force_Fx = msg.force_Fx;
    }
    else {
      resolved.force_Fx = 0.0
    }

    if (msg.force_Fy !== undefined) {
      resolved.force_Fy = msg.force_Fy;
    }
    else {
      resolved.force_Fy = 0.0
    }

    if (msg.force_Fz !== undefined) {
      resolved.force_Fz = msg.force_Fz;
    }
    else {
      resolved.force_Fz = 0.0
    }

    if (msg.force_Mx !== undefined) {
      resolved.force_Mx = msg.force_Mx;
    }
    else {
      resolved.force_Mx = 0.0
    }

    if (msg.force_My !== undefined) {
      resolved.force_My = msg.force_My;
    }
    else {
      resolved.force_My = 0.0
    }

    if (msg.force_Mz !== undefined) {
      resolved.force_Mz = msg.force_Mz;
    }
    else {
      resolved.force_Mz = 0.0
    }

    return resolved;
    }
};

module.exports = Six_Force;
