import React, { useState, useEffect } from "react";
import io from "socket.io-client";

// Connect to the backend 
const socket = io("http://127.0.0.1:5000");

const AttendanceDashboard = () => {
  const [attendance, setAttendance] = useState([]);

  useEffect(() => {
    // Fetch initial attendance records
    fetch("http://127.0.0.1:5000/attendance")
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        setAttendance(data)
      });
    socket.on("connect", () => {
      console.log("Connected to Socket.IO server!");
    });



  }, []);


  useEffect(() => {
    // Listen for real-time updates
    socket.on("new_attendance", (record) => {
      record = record.data;
      setAttendance((prevAttendance) => {
        // Check if the student_id already exists in the attendance array
        const isDuplicate = prevAttendance.some(
          (entry) => entry[0] === record.student_id
        );

        // If it's not a duplicate, add the new record
        if (!isDuplicate) {
          return [...prevAttendance, [record.student_id, record.student_name, record.timestamp]];
        }

        // Otherwise, return the previous state unchanged
        return prevAttendance;
      });
    });

    // Clean up the event listener when the component unmounts
    return () => {
      socket.off("new_attendance");
    };
  }, [socket]);


  return (
    <div style={{ padding: "20px" }}>
      <h1>Attendance Dashboard</h1>
      <table border="1" style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th>Student ID</th>
            <th>Name</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {attendance && attendance.map((record, index) => (
            <tr key={index}>
              <td>{record[0]}</td>
              <td>{record[1]}</td>
              <td>{record[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AttendanceDashboard;
