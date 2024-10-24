// Generated by gencpp from file rm_msgs/Joint_Error_Code.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_JOINT_ERROR_CODE_H
#define RM_MSGS_MESSAGE_JOINT_ERROR_CODE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rm_msgs
{
template <class ContainerAllocator>
struct Joint_Error_Code_
{
  typedef Joint_Error_Code_<ContainerAllocator> Type;

  Joint_Error_Code_()
    : joint_error()  {
      joint_error.assign(0);
  }
  Joint_Error_Code_(const ContainerAllocator& _alloc)
    : joint_error()  {
  (void)_alloc;
      joint_error.assign(0);
  }



   typedef boost::array<uint16_t, 6>  _joint_error_type;
  _joint_error_type joint_error;





  typedef boost::shared_ptr< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> const> ConstPtr;

}; // struct Joint_Error_Code_

typedef ::rm_msgs::Joint_Error_Code_<std::allocator<void> > Joint_Error_Code;

typedef boost::shared_ptr< ::rm_msgs::Joint_Error_Code > Joint_Error_CodePtr;
typedef boost::shared_ptr< ::rm_msgs::Joint_Error_Code const> Joint_Error_CodeConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::Joint_Error_Code_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::Joint_Error_Code_<ContainerAllocator1> & lhs, const ::rm_msgs::Joint_Error_Code_<ContainerAllocator2> & rhs)
{
  return lhs.joint_error == rhs.joint_error;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::Joint_Error_Code_<ContainerAllocator1> & lhs, const ::rm_msgs::Joint_Error_Code_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
{
  static const char* value()
  {
    return "74ddce861d3ff625b60dae7918fad457";
  }

  static const char* value(const ::rm_msgs::Joint_Error_Code_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x74ddce861d3ff625ULL;
  static const uint64_t static_value2 = 0xb60dae7918fad457ULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/Joint_Error_Code";
  }

  static const char* value(const ::rm_msgs::Joint_Error_Code_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint16[6] joint_error   #每个关节报错信息\n"
;
  }

  static const char* value(const ::rm_msgs::Joint_Error_Code_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joint_error);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Joint_Error_Code_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::Joint_Error_Code_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::Joint_Error_Code_<ContainerAllocator>& v)
  {
    s << indent << "joint_error[]" << std::endl;
    for (size_t i = 0; i < v.joint_error.size(); ++i)
    {
      s << indent << "  joint_error[" << i << "]: ";
      Printer<uint16_t>::stream(s, indent + "  ", v.joint_error[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_JOINT_ERROR_CODE_H
