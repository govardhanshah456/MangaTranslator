package com.mangatranslator.mangatranslator.models;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "documents")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Document {
    @Id
    private String id;

    @Column(nullable = false)
    private String originalFilename;

    @Column(nullable = false)
    private String inputPath;

    private String mimeType;

    @Column(nullable = false)
    private Long sizeBytes;

    @Column(nullable = false)
    private LocalDateTime createdAt;
}
