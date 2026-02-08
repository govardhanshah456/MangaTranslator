package com.mangatranslator.mangatranslator.models;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity()
@Table(
        name = "jobs",
        indexes = {
                @Index(name = "idx_jobs_document_id", columnList = "documentId")
        }
)@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Job {
    @Id
    private String id;

    @Column(nullable = false)
    private String documentId;

    @Column(nullable = false)
    private String sourceLang;

    @Column(nullable = false)
    private String targetLang;

    @Column(nullable = false)
    private String status;

    @Column(nullable = false)
    private Integer progress;

    private String outputPath;

    private String errorMessage;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column
    private  String ocrOutputPath;
}
